$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ReferencePath = "C:\Users\Administrator\Desktop\论文集合\mi\说明文档.docx"
$OutputPath = "E:\世界机器人大赛\projects\arm_bci_robot\docs\MI机械臂控制方法说明_2026-07-25.docx"

if (-not (Test-Path -LiteralPath $ReferencePath)) {
    throw "参考文档不存在：$ReferencePath"
}

Copy-Item -LiteralPath $ReferencePath -Destination $OutputPath -Force

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

function Set-BodyFormat($selection) {
    $selection.Font.NameFarEast = "宋体"
    $selection.Font.Name = "Times New Roman"
    $selection.Font.Size = 12
    $selection.Font.Bold = 0
    $selection.Font.Italic = 0
    $selection.ParagraphFormat.Alignment = 3
    $selection.ParagraphFormat.FirstLineIndent = 24
    $selection.ParagraphFormat.SpaceBefore = 0
    $selection.ParagraphFormat.SpaceAfter = 0
    $selection.ParagraphFormat.LineSpacingRule = 1
    $selection.ParagraphFormat.LineSpacing = 18
    $selection.ParagraphFormat.KeepTogether = 0
    $selection.ParagraphFormat.KeepWithNext = 0
    $selection.ParagraphFormat.PageBreakBefore = 0
    $selection.ParagraphFormat.WidowControl = -1
    $selection.Range.HighlightColorIndex = 0
    $selection.Range.Shading.BackgroundPatternColor = -16777216
}

function Add-Body($selection, [string]$text) {
    try { $selection.Style = "正文" } catch {}
    Set-BodyFormat $selection
    $selection.TypeText($text)
    $selection.TypeParagraph()
}

function Add-Title($selection, [string]$text) {
    try { $selection.Style = "标题" } catch {}
    $selection.Font.NameFarEast = "微软雅黑"
    $selection.Font.Name = "Arial"
    $selection.Font.Size = 18
    $selection.Font.Bold = -1
    $selection.ParagraphFormat.Alignment = 1
    $selection.ParagraphFormat.FirstLineIndent = 0
    $selection.ParagraphFormat.SpaceBefore = 12
    $selection.ParagraphFormat.SpaceAfter = 0
    $selection.ParagraphFormat.LineSpacingRule = 1
    $selection.ParagraphFormat.LineSpacing = 18
    $selection.TypeText($text)
    $selection.TypeParagraph()
}

function Add-Subtitle($selection, [string]$text) {
    try { $selection.Style = "正文" } catch {}
    $selection.Font.NameFarEast = "黑体"
    $selection.Font.Name = "Arial"
    $selection.Font.Size = 14
    $selection.Font.Bold = 0
    $selection.ParagraphFormat.Alignment = 1
    $selection.ParagraphFormat.FirstLineIndent = 0
    $selection.ParagraphFormat.SpaceBefore = 0
    $selection.ParagraphFormat.SpaceAfter = 0
    $selection.ParagraphFormat.LineSpacingRule = 1
    $selection.ParagraphFormat.LineSpacing = 18
    $selection.TypeText($text)
    $selection.TypeParagraph()
}

function Add-H1($selection, [string]$text) {
    try { $selection.Style = "标题 1" } catch {}
    $selection.Font.NameFarEast = "微软雅黑"
    $selection.Font.Name = "Arial"
    $selection.Font.Size = 15
    $selection.Font.Bold = -1
    $selection.Font.Color = 0
    $selection.ParagraphFormat.Alignment = 0
    $selection.ParagraphFormat.FirstLineIndent = 0
    $selection.ParagraphFormat.SpaceBefore = 18
    $selection.ParagraphFormat.SpaceAfter = 0
    $selection.ParagraphFormat.LineSpacingRule = 1
    $selection.ParagraphFormat.LineSpacing = 18
    $selection.ParagraphFormat.KeepTogether = -1
    $selection.ParagraphFormat.KeepWithNext = -1
    $selection.ParagraphFormat.PageBreakBefore = 0
    $selection.Range.HighlightColorIndex = 0
    $selection.Range.Shading.BackgroundPatternColor = -16777216
    $selection.TypeText($text)
    $selection.TypeParagraph()
}

function Add-H2($selection, [string]$text) {
    try { $selection.Style = "标题 2" } catch {}
    $selection.Font.NameFarEast = "宋体"
    $selection.Font.Name = "Times New Roman"
    $selection.Font.Size = 13
    $selection.Font.Bold = -1
    $selection.Font.Color = 0
    $selection.ParagraphFormat.Alignment = 0
    $selection.ParagraphFormat.FirstLineIndent = 0
    $selection.ParagraphFormat.SpaceBefore = 11
    $selection.ParagraphFormat.SpaceAfter = 6
    $selection.ParagraphFormat.LineSpacingRule = 1
    $selection.ParagraphFormat.LineSpacing = 18
    $selection.ParagraphFormat.KeepTogether = -1
    $selection.ParagraphFormat.KeepWithNext = -1
    $selection.ParagraphFormat.PageBreakBefore = 0
    $selection.Range.HighlightColorIndex = 0
    $selection.Range.Shading.BackgroundPatternColor = -16777216
    $selection.TypeText($text)
    $selection.TypeParagraph()
}

function Add-Caption($selection, [string]$text) {
    try { $selection.Style = "题注" } catch {}
    $selection.Font.NameFarEast = "宋体"
    $selection.Font.Name = "Times New Roman"
    $selection.Font.Size = 10
    $selection.Font.Bold = -1
    $selection.ParagraphFormat.Alignment = 1
    $selection.ParagraphFormat.FirstLineIndent = 0
    $selection.ParagraphFormat.SpaceBefore = 0
    $selection.ParagraphFormat.SpaceAfter = 8
    $selection.ParagraphFormat.LineSpacingRule = 1
    $selection.ParagraphFormat.LineSpacing = 18
    $selection.TypeText($text)
    $selection.TypeParagraph()
}

function Add-Table($document, $selection, [object[]]$rows, [double[]]$widths = @()) {
    if ($rows.Count -lt 1) { return }
    $columnCount = $rows[0].Count
    $range = $selection.Range
    $table = $document.Tables.Add($range, $rows.Count, $columnCount)
    $table.Borders.Enable = 1
    $table.Rows.Alignment = 1
    $table.AllowAutoFit = $true
    $table.AutoFitBehavior(2)
    $table.Range.Font.NameFarEast = "宋体"
    $table.Range.Font.Name = "Times New Roman"
    $table.Range.Font.Size = 12
    $table.Range.ParagraphFormat.FirstLineIndent = 0
    $table.Range.ParagraphFormat.SpaceBefore = 0
    $table.Range.ParagraphFormat.SpaceAfter = 0
    $table.Range.ParagraphFormat.LineSpacingRule = 0
    $table.Range.ParagraphFormat.LineSpacing = 12
    for ($r = 0; $r -lt $rows.Count; $r++) {
        for ($c = 0; $c -lt $columnCount; $c++) {
            $cell = $table.Cell($r + 1, $c + 1)
            $cell.Range.Text = [string]$rows[$r][$c]
            $cell.VerticalAlignment = 1
            if ($r -eq 0) {
                $cell.Range.Bold = -1
                $cell.Range.ParagraphFormat.Alignment = 1
            }
            elseif ($c -eq 0) {
                $cell.Range.ParagraphFormat.Alignment = 1
            }
            else {
                $cell.Range.ParagraphFormat.Alignment = 0
            }
        }
    }
    $table.Rows.Item(1).HeadingFormat = -1
    if ($widths.Count -eq $columnCount) {
        for ($c = 0; $c -lt $columnCount; $c++) {
            $table.Columns.Item($c + 1).PreferredWidthType = 2
            $table.Columns.Item($c + 1).PreferredWidth = $widths[$c]
        }
    }
    $selection.SetRange($table.Range.End, $table.Range.End)
    $selection.TypeParagraph()
    Set-BodyFormat $selection
}

try {
    $doc = $word.Documents.Open($OutputPath, $false, $false)

    # 保留参考文档的样式定义和页面设置，仅清除正文及所有原浮动对象。
    $doc.Content.Delete()
    for ($shapeIndex = $doc.Shapes.Count; $shapeIndex -ge 1; $shapeIndex--) {
        $doc.Shapes.Item($shapeIndex).Delete()
    }
    for ($inlineIndex = $doc.InlineShapes.Count; $inlineIndex -ge 1; $inlineIndex--) {
        $doc.InlineShapes.Item($inlineIndex).Delete()
    }
    foreach ($section in $doc.Sections) {
        foreach ($header in $section.Headers) { $header.Range.Text = "" }
        foreach ($footer in $section.Footers) { $footer.Range.Text = "" }
    }
    $selection = $word.Selection
    $selection.HomeKey(6) | Out-Null

    Add-Title $selection "MI机械臂控制方法说明报告"
    Add-Subtitle $selection "项目名称：基于左/右手运动想象编码的多模态机械臂控制系统"
    Add-Subtitle $selection "版本：1.0　日期：2026年7月25日"

    Add-H1 $selection "一、方法概述"
    Add-Body $selection "本方法是在现有机械臂脑机接口系统基础上新增的运动想象（motor imagery，MI）控制方案。系统只采集并分类左手运动想象和右手运动想象，将左手分类结果记为二进制1，将右手分类结果记为二进制0，再通过1～3个分类窗口组成可变长二进制编码。眼电用于静息态与运动态的双向切换，左右咬肌肌电用于提交、回退、确认和紧急停止。该方法只新增解码层，不替换原有机械臂通信、速度参数、动作执行、确认窗口和TCP输出逻辑。"
    Add-Body $selection "方法设计目标是以左/右手MI分类形成八类机械臂意图，并通过用户可主动触发的咬肌编辑机制降低连续误分类造成的风险。六个方向指令沿用原持续运动逻辑，抓取和放置沿用原确认窗口，双侧咬肌作为最高优先级的实时停止输入。"

    Add-H1 $selection "二、系统组成与通道配置"
    Add-H2 $selection "2.1信号组成"
    Add-Body $selection "系统输入由32导EEG和辅助生理信号组成。EEG用于左、右手MI分类；眼电用于三次眨眼状态切换；左、右咬肌肌电用于序列提交、撤销和紧急停止。采集端通过BrainVision LSL Connector统一接收连续信号、采样时间戳和事件标记，当前采样率为250 Hz。"
    Add-H2 $selection "2.2已确认的32导物理顺序"
    Add-Body $selection "根据BrainVision通道列表和电极位置图，32导EEG物理顺序见表1。C3、Cz和C4分别对应第8、24和25路；Gnd为44，Ref为45，Gnd和Ref不作为32路分类输入。"
    Add-Caption $selection "表1  32导EEG物理通道顺序"
    $channelRows = @(
        @("编号", "通道", "编号", "通道", "编号", "通道", "编号", "通道"),
        @("1","Fp1","9","T7","17","Oz","25","C4"),
        @("2","Fz","10","TP9","18","O2","26","T8"),
        @("3","F3","11","CP5","19","P4","27","FT10"),
        @("4","F7","12","CP1","20","P8","28","FC6"),
        @("5","FT9","13","Pz","21","TP10","29","FC2"),
        @("6","FC5","14","P3","22","CP6","30","F4"),
        @("7","FC1","15","P7","23","CP2","31","F8"),
        @("8","C3","16","O1","24","Cz","32","Fp2")
    )
    Add-Table $doc $selection $channelRows
    Add-Body $selection "训练和在线控制必须保持完全相同的物理通道顺序。当前LSL标签仅显示EEG 1_1～EEG 1_32，因此系统后续应将表1中的真实通道名称写入配置和采集文件元数据，避免通道显示、坏道定位、C3/C4分析和拓扑图解释错误。"

    Add-H1 $selection "三、MI采集范式"
    Add-H2 $selection "3.1范式选择"
    Add-Body $selection "在准备配置页面的采集/在线范式选项中选择MI（左手=1、右手=0）。选择后，训练采集和在线控制统一使用该范式；原有其他范式继续保留，各范式互不替代。MI采集只生成C1左手运动想象和C2右手运动想象两类标签。"
    Add-H2 $selection "3.2任务定义"
    Add-Caption $selection "表2  MI训练任务定义"
    $taskRows = @(
        @("类别", "任务", "位值", "操作要求"),
        @("C1", "左手运动想象", "1", "想象左手反复握拳、松开，感受手指屈伸和掌心收紧，不真实移动手和肩膀。"),
        @("C2", "右手运动想象", "0", "想象右手反复握拳、松开，保持与左手相同的节奏、力度想象和注意方式。")
    )
    Add-Table $doc $selection $taskRows
    Add-H2 $selection "3.3单个trial时间结构"
    Add-Body $selection "近期MI记录采用4 s静息基线、2 s任务提示和2 s运动想象阶段。具体时长由配置页控制，训练与离线分析以采集文件中保存的baseline_start_ms、cue_onset_ms、task_start_ms和task_end_ms为准。每个session内左右手trial数量应保持均衡并随机排列。"
    Add-Body $selection "建议每类累计采集60～100个有效trial，并拆分为多个独立session。采集过程中应避免真实手部动作、肩部动作、咬牙和规律性眼动，保证模型学习运动想象脑电而非外周肌电或视觉提示差异。"

    Add-H1 $selection "四、EEG数据处理与模型训练"
    Add-H2 $selection "4.1训练与在线共用预处理"
    Add-Body $selection "每个在线分类窗口长度为2 s，在250 Hz采样率下包含500个采样点。训练、离线测试和在线预测共用同一特征处理对象，依次执行全通道平均参考（CAR）、50 Hz陷波（Q=30）、1～45 Hz带通、逐通道逐窗口标准化，然后对8～12、12～16、16～20、20～24、24～30和30～40 Hz六个子频带进行滤波。滤波采用双向零相位处理，以避免引入相位偏移。"
    Add-Body $selection "每个子频带提取通道对数方差或功率特征；运行环境支持时加入带Ledoit-Wolf正则化的共同空间模式（CSP）特征。CSP用于增强两类运动想象之间的空间方差差异，是左右手MI分类的经典方法。"
    Add-H2 $selection "4.2候选模型自动比较"
    Add-Caption $selection "表3  训练阶段候选模型"
    $modelRows = @(
        @("序号", "候选模型", "说明"),
        @("1", "Soft Voting", "融合LDA、Logistic Regression、SVM及树模型概率"),
        @("2", "LDA shrinkage", "适用于小样本、高维EEG特征"),
        @("3", "Logistic Regression", "线性概率分类基线"),
        @("4", "RBF-SVM", "非线性分类候选"),
        @("5", "ExtraTrees", "极端随机树集成"),
        @("6", "RandomForest", "随机森林集成"),
        @("7", "GaussianNB", "低复杂度概率分类基线")
    )
    Add-Table $doc $selection $modelRows
    Add-Body $selection "点击开始训练按钮后，系统依次训练和验证表3中的全部模型，并按照交叉验证准确率、Kappa系数和最低类别准确率形成排名。训练完成后自动保存最佳模型到统一在线模型路径，在线测试加载同一模型对象、信号配置和权重。"
    Add-H2 $selection "4.3验证与数据泄露控制"
    Add-Body $selection "交叉验证以trial为划分单位，同一trial产生的窗口不得同时出现在训练集和验证集中。独立测试记录不得同时勾选为训练数据。除K折交叉验证外，应报告跨session留一测试、混淆矩阵、最低类别召回率和标签置换检验结果；只有跨session结果稳定高于随机水平，才可认为模型具备在线泛化能力。"

    Add-H1 $selection "五、二进制编码与机械臂指令"
    Add-Body $selection "左、右手运动想象分类结果按时间顺序写入位缓存。编码长度由用户提交时机决定，支持1位、2位和3位指令。编码与机械臂动作的对应关系见表4。"
    Add-Caption $selection "表4  MI编码与机械臂动作映射"
    $codeRows = @(
        @("编码", "分类序列", "机械臂意图", "执行方式"),
        @("1", "左手MI", "左移", "持续运动"),
        @("0", "右手MI", "右移", "持续运动"),
        @("10", "左手MI→右手MI", "前进", "持续运动"),
        @("01", "右手MI→左手MI", "后退", "持续运动"),
        @("11", "左手MI→左手MI", "上升", "持续运动"),
        @("00", "右手MI→右手MI", "下降", "持续运动"),
        @("xx1", "任意前两位→左手MI", "抓取", "进入确认窗口"),
        @("xx0", "任意前两位→右手MI", "放置/释放", "进入确认窗口")
    )
    Add-Table $doc $selection $codeRows
    Add-Body $selection "xx表示前两位可以是00、01、10或11中的任意组合，第三位决定特殊动作。三位编码结束后必须通过左咬肌提交；如果在下一窗口仍未检测到咬肌，则默认取消该组编码并清空当前全部计数，界面仅显示未执行咬肌，不发送机械臂命令。"

    Add-H1 $selection "六、在线状态机与实时解码"
    Add-H2 $selection "6.1静息态与运动态"
    Add-Body $selection "MI在线控制启动后默认进入静息态。静息态只检测眼电，不接收左右手MI分类、咬肌编码或机械臂动作输出。检测到三次眨眼后切换至运动态，并从第一个MI分类窗口开始计数。运动态再次检测到三次眨眼时切换回静息态、清空当前编码；如果机械臂正在持续运动，系统同时发送stop_motion。"
    Add-H2 $selection "6.2在线分类窗口"
    Add-Body $selection "运动态每个2 s EEG窗口输出左手和右手分类概率、最高类别置信度及Margin。无伪迹且无咬肌事件时，左手结果写入1，右手结果写入0。伪迹窗口可以显示模型预测，但不写入位缓存。"
    Add-H2 $selection "6.3事件优先级"
    Add-Caption $selection "表5  在线事件处理优先级"
    $priorityRows = @(
        @("优先级", "事件", "系统处理"),
        @("1", "状态切换眨眼达到阈值", "切换静息态/运动态并清空当前序列"),
        @("2", "双侧咬肌", "立即停止机械臂并丢弃当前EEG窗口"),
        @("3", "单侧咬肌", "立即提交或回退，丢弃当前EEG窗口"),
        @("4", "EEG/AUX伪迹", "拒绝当前窗口计数"),
        @("5", "正常EEG结果", "将1或0写入编码缓存")
    )
    Add-Table $doc $selection $priorityRows

    Add-H1 $selection "七、咬肌编辑、确认与持续运动"
    Add-H2 $selection "7.1序列编辑"
    Add-Caption $selection "表6  咬肌事件语义"
    $jawRows = @(
        @("输入", "语义", "当前窗口", "缓存变化"),
        @("左咬肌", "提交已完成编码", "立即终止并丢弃，不参与提交", "提交后清空"),
        @("右咬肌", "回退一个已完成窗口", "立即终止并丢弃", "删除最后一位"),
        @("双侧咬肌", "迅速停止", "立即终止并丢弃", "清空全部编码"),
        @("三位后未执行咬肌", "默认取消", "不进入下一组", "清空目前计数")
    )
    Add-Table $doc $selection $jawRows
    Add-Body $selection "左咬肌提交的是咬肌发生前已经完成的窗口，不包含当前尚未结束的窗口。例如缓存已完成10，第三窗口中检测到左咬肌，则立即提交10，第三窗口即使已有临时预测也不保存。右咬肌不是清空全部编码，而是撤销最近一个已完成窗口并回退一位。"
    Add-H2 $selection "7.2抓取与放置确认"
    Add-Body $selection "提交xx1或xx0后进入原系统确认窗口。确认窗口只确认已经解码的抓取或放置动作，不重新进行EEG分类。确认时间内咬肌计数达到配置阈值时执行动作；未达到阈值时取消，不发送机械臂命令。"
    Add-H2 $selection "7.3持续运动与安全停止"
    Add-Body $selection "左移、右移、前进、后退、上升和下降沿用原持续运动输出逻辑与速度参数；抓取和放置为离散动作。MI运动态中，双侧咬肌取代旧的三次眨眼急停方式，作为实时迅速停止输入。停止事件优先于提交、回退、分类结果和确认窗口，并复用原机械臂命令总线及TCP协议。"

    Add-H1 $selection "八、数据保存、在线模型与质量控制"
    Add-H2 $selection "8.1数据与模型契约"
    Add-Body $selection "每份采集文件保存32导EEG、AUX、完整原始连续流、采样率、原始时间戳、trial标签、基线/提示/任务事件、范式配置、完成状态和session标识。模型文件保存信号配置、2 s窗口长度、50 Hz陷波参数、滤波器组、特征变换、分类器及类别顺序。在线测试必须加载训练完成后自动选出的最佳模型，且采样率、通道数和通道顺序必须与训练一致。"
    Add-H2 $selection "8.2采集后质量检查"
    Add-Body $selection "每次采集后应首先检查采样率、通道数量、记录时长、事件数量和左右标签平衡，再检查平线、非有限值、异常方差、电极跳变、50 Hz工频和宽带肌电。完成CAR、陷波和带通后，应重点分析C3、C4及FC/CP邻近运动区的μ节律（8～13 Hz）和β节律（13～30 Hz）事件相关去同步化（ERD）。"
    Add-Body $selection "模型准确率高并不自动证明存在稳定脑信号。正式评价应同时满足：跨session准确率显著高于随机水平、标签置换检验通过、左右手类别召回率均衡、C3/C4侧化方向在不同session中可重复。当前新采数据尚未显示稳定的C3/C4左右手侧化，因此真实机械臂测试应保留低速、人工急停和双侧咬肌急停。"

    Add-H1 $selection "九、标准操作流程"
    Add-Body $selection "（1）在准备配置中选择MI范式；（2）核对32导物理顺序、C3/C4位置、Ref/Gnd和AUX贴片；（3）采集均衡的左右手运动想象数据；（4）在模型训练页勾选训练记录并运行全部候选模型；（5）查看完整排名、Kappa、最低类别准确率和独立测试结果；（6）确认最佳模型已同步到在线模型路径；（7）在线启动后保持静息，三次眨眼进入运动态；（8）通过左右手MI形成编码，使用左咬肌提交、右咬肌回退、双侧咬肌急停；（9）抓取和放置按确认窗口完成二次确认；（10）测试结束后停止在线控制并保存日志。"

    # 参考模板的标题样式带红色段落下边框；本报告只保留其字体、
    # 字号、间距和层级，去除所有正文段落边框。表格边框不受影响。
    foreach ($paragraph in $doc.Paragraphs) {
        $paragraph.Borders.Enable = 0
    }

    # 全文检查并统一中文正文基础格式；标题、题注和表格保留各自设置。
    $doc.Fields.Update() | Out-Null
    $doc.Save()
    $doc.Close($true)
}
finally {
    try { $word.Quit() } catch {}
    try { [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null } catch {}
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Get-Item -LiteralPath $OutputPath | Select-Object FullName, Length, LastWriteTime







