# -*- coding: utf-8 -*-
"""Command implementations for the arm BCI system."""

from __future__ import annotations

from .core import *

def build_robot(args) -> RobotArmClient:
    if getattr(args, "robot_host", None):
        return SocketRobotArmClient(args.robot_host, args.robot_port, timeout=args.robot_timeout)
    return MockRobotArm(action_delay=getattr(args, "mock_delay", 0.15))


def apply_runtime_control_config(model: HybridBCIModel, args=None) -> None:
    cfg = model.decoder_config
    if not hasattr(cfg, "refractory_seconds"):
        cfg.refractory_seconds = 0.25
    if not hasattr(cfg, "command_duration_seconds"):
        cfg.command_duration_seconds = 0.55
    if args is not None and getattr(args, "refractory_seconds", None) is not None:
        cfg.refractory_seconds = float(args.refractory_seconds)
    if args is not None and getattr(args, "command_duration", None) is not None:
        cfg.command_duration_seconds = float(args.command_duration)


def print_metrics(metrics: Dict) -> None:
    if not metrics:
        print("No validation metrics were produced; provide at least two trials per class.")
        return
    print(json.dumps(metrics_summary(metrics), indent=2, ensure_ascii=False))


def metrics_summary(metrics: Dict) -> Dict:
    keys = [
        "window_count",
        "eeg_only_accuracy",
        "eeg_only_kappa",
        "aux_only_accuracy",
        "aux_only_kappa",
        "fusion_accuracy",
        "fusion_kappa",
        "eeg_weight",
    ]
    return {k: metrics[k] for k in keys if k in metrics}


def command_train(args) -> None:
    cfg = SignalConfig(fs=args.fs)
    decoder_cfg = DecoderConfig(
        eeg_weight=args.eeg_weight,
        command_confidence=args.command_confidence,
        stable_windows=args.stable_windows,
        refractory_seconds=args.refractory_seconds,
        command_duration_seconds=args.command_duration,
    )
    eeg, y, aux, file_fs = load_calibration_npz(args.data)
    if file_fs:
        cfg.fs = file_fs
    cfg.eeg_channels = load_npz_channel_names(args.data, ensure_3d_epochs(eeg, "eeg").shape[1])
    cfg.use_csp = not args.no_csp

    model = HybridBCIModel(cfg, decoder_cfg)
    metrics = model.fit(eeg, y, aux=aux, classifier_kind=args.classifier)
    model.save(args.model)

    report_path = os.path.splitext(args.model)[0] + "_report.json"
    report = {
        "model": args.model,
        "data": args.data,
        "signal_config": asdict(cfg),
        "decoder_config": asdict(decoder_cfg),
        "metrics": metrics,
        "paradigm": MI8_PARADIGM,
        "competition_notes": {
            "hybrid_paradigm": "Six movement commands are decoded from 3 masseter patterns times 2 MI hand-imagery patterns.",
            "grip_state": "Grab/release is not a model class; a triple-blink event toggles the gripper state.",
            "direct_control": "The decoder predicts rest plus 6 movement commands; rest is a no-command safety state.",
            "false_trigger_suppression": "Every command passes confidence, margin, stability, refractory and artifact gates before execution.",
            "online_latency_target": f"window={cfg.window_seconds}s, step={cfg.step_seconds}s, stable_windows={decoder_cfg.stable_windows}",
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved model: {args.model}")
    print(f"Saved report: {report_path}")
    print_metrics(metrics)

    eeg_acc = metrics.get("eeg_only_accuracy")
    aux_acc = metrics.get("aux_only_accuracy")
    if eeg_acc is not None and eeg_acc < 0.60:
        print("WARNING: EEG-only accuracy is below the competition baseline target of 60%.")
    if eeg_acc is not None and aux_acc is not None and eeg_acc <= aux_acc:
        print("WARNING: EEG-only accuracy is not higher than auxiliary-only accuracy; improve EEG features/calibration.")


def command_evaluate(args) -> None:
    model = HybridBCIModel.load(args.model)
    eeg, y, aux, _file_fs = load_calibration_npz(args.data)
    y_idx = labels_to_indices(y)
    eeg_windows, y_windows, aux_windows = make_windows_from_trials(eeg, y_idx, model.signal_config, aux)
    metrics = model.evaluate_windows(eeg_windows, y_windows, aux_windows)
    print_metrics(metrics)


def command_make_demo_data(args) -> None:
    cfg = SignalConfig(fs=args.fs)
    eeg, y, aux = build_demo_calibration(cfg, trials_per_class=args.trials_per_class)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        eeg=eeg,
        y=y,
        aux=aux,
        fs=cfg.fs,
        channel_names=np.asarray(cfg.eeg_channels, dtype=object),
        aux_channel_names=np.asarray(["LeftMasseter", "RightMasseter", "VirtualBlink"], dtype=object),
        paradigm_name=PARADIGM_NAME,
        paradigm_json=json.dumps(MI8_PARADIGM, ensure_ascii=False),
        mode="simulated",
    )
    print(f"Saved demo calibration data: {args.out}")


def command_simulate(args) -> None:
    if args.model and os.path.exists(args.model):
        model = HybridBCIModel.load(args.model)
    else:
        cfg = SignalConfig(fs=args.fs)
        eeg, y, aux = build_demo_calibration(cfg, trials_per_class=args.trials_per_class)
        model = HybridBCIModel(cfg, DecoderConfig())
        metrics = model.fit(eeg, y, aux)
        if args.model:
            model.save(args.model)
            print(f"Saved simulated model: {args.model}")
        print_metrics(metrics)

    apply_runtime_control_config(model, args)
    robot = build_robot(args)
    controller = AsyncIntentController(model, robot, target=args.target)
    grip_controller = GripStateController(
        robot,
        target=args.target,
        duration_seconds=args.command_duration,
    )
    source = SimulatedSignalSource(model.signal_config, cycles=args.cycles)

    accepted = 0
    grip_toggles = 0
    total = 0
    while True:
        try:
            eeg, aux, now, true_label = source.read_window()
        except StopIteration:
            break
        if is_grip_toggle_event(true_label):
            command = grip_controller.toggle(now=now, confidence=1.0)
            total += 1
            if command is not None:
                grip_toggles += 1
                accepted += 1
                print(
                    f"[BLINK] true={true_label:9s} grip_state={grip_controller.state_text} "
                    f"cmd={command.name} seq={command.sequence_id}"
                )
            if args.realtime:
                time.sleep(model.signal_config.step_seconds)
            continue
        decision = controller.update(eeg, aux, now=now)
        total += 1
        if decision.accepted:
            accepted += 1
        print(
            f"[ONLINE] true={true_label:9s} pred={decision.intent:9s} "
            f"conf={decision.confidence:.3f} margin={decision.margin:.3f} "
            f"state={decision.state:11s} accepted={decision.accepted} reason={decision.reason}"
        )
        if args.realtime:
            time.sleep(model.signal_config.step_seconds)

    robot.close()
    print(f"Simulation complete: windows={total}, accepted_commands={accepted}, grip_toggles={grip_toggles}")


def command_online(args) -> None:
    model = HybridBCIModel.load(args.model)
    apply_runtime_control_config(model, args)
    robot = build_robot(args)
    controller = AsyncIntentController(model, robot, target=args.target)
    grip_controller = GripStateController(
        robot,
        target=args.target,
        duration_seconds=args.command_duration,
    )

    if args.rda:
        source = RDA64OnlineSource(
            model.signal_config,
            host=args.rda_host,
            port=args.rda_port,
            expected_channels=args.channels,
            aux_channels=args.aux_channels,
            log_callback=print,
        )
        source.start()
    elif args.replay:
        source = NpzReplaySource(args.replay, model.signal_config)
    else:
        raise RuntimeError(
            "No online source is configured. Use --replay for an npz replay stream or --rda for BrainVision RDA."
        )

    try:
        while True:
            try:
                eeg, aux, now, label = source.read_window()
            except BlockingIOError:
                time.sleep(0.02)
                continue
            except StopIteration:
                break
            if is_grip_toggle_event(label):
                command = grip_controller.toggle(now=now, confidence=1.0)
                payload = {
                    "event": GRIP_TOGGLE_LABEL,
                    "grip_state": grip_controller.state_text,
                    "command": asdict(command) if command is not None else None,
                    "accepted": command is not None,
                }
                print(json.dumps(payload, ensure_ascii=False))
                if args.realtime:
                    time.sleep(model.signal_config.step_seconds)
                continue
            decision = controller.update(eeg, aux, now=now)
            print(json.dumps(asdict(decision), ensure_ascii=False))
            if args.realtime:
                time.sleep(model.signal_config.step_seconds)
    finally:
        if hasattr(source, "close"):
            source.close()
        robot.close()


