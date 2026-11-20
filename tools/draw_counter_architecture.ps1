Add-Type -AssemblyName System.Drawing

$out = Join-Path $PSScriptRoot '..\docs\emg_counter_window_flow.png'
$out = [System.IO.Path]::GetFullPath($out)
$width = 1800
$height = 1120
$bmp = New-Object System.Drawing.Bitmap($width, $height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g.Clear([System.Drawing.Color]::FromArgb(248, 251, 253))

$font = New-Object System.Drawing.Font('Microsoft YaHei UI', 18)
$small = New-Object System.Drawing.Font('Microsoft YaHei UI', 14)
$bold = New-Object System.Drawing.Font('Microsoft YaHei UI', 24, [System.Drawing.FontStyle]::Bold)
$title = New-Object System.Drawing.Font('Microsoft YaHei UI', 32, [System.Drawing.FontStyle]::Bold)
$ink = [System.Drawing.Color]::FromArgb(17, 52, 77)
$muted = [System.Drawing.Color]::FromArgb(75, 100, 120)

function Box([int]$x,[int]$y,[int]$w,[int]$h,[string]$text,[System.Drawing.Color]$fill,[System.Drawing.Color]$border,[System.Drawing.Font]$useFont=$font) {
  $rect = New-Object System.Drawing.Rectangle($x,$y,$w,$h)
  $brush = New-Object System.Drawing.SolidBrush($fill)
  $pen = New-Object System.Drawing.Pen($border, 2)
  $g.FillRectangle($brush,$rect); $g.DrawRectangle($pen,$rect)
  $fmt = New-Object System.Drawing.StringFormat
  $fmt.Alignment = [System.Drawing.StringAlignment]::Center; $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center
  $textRect = New-Object System.Drawing.RectangleF([single]$x,[single]$y,[single]$w,[single]$h)
  $g.DrawString($text,$useFont,(New-Object System.Drawing.SolidBrush($ink)),$textRect,$fmt)
  $brush.Dispose(); $pen.Dispose(); $fmt.Dispose()
}
function Arrow([int]$x1,[int]$y1,[int]$x2,[int]$y2,[string]$label='') {
  $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(8,145,178),4)
  $pen.CustomEndCap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(6,7,$true)
  $g.DrawLine($pen,$x1,$y1,$x2,$y2)
  if ($label) { $g.DrawString($label,$small,(New-Object System.Drawing.SolidBrush($muted)),(($x1+$x2)/2-35),(($y1+$y2)/2-24)) }
  $pen.Dispose()
}

$g.DrawString('肌电事件统计与窗口计数器（统一当前窗口）',$title,(New-Object System.Drawing.SolidBrush($ink)),70,28)
$g.DrawString('原则：窗口切换时统一清零；每次只检测最新 180 ms；仅“未激活 → 激活”的上升沿使计数 +1。',$font,(New-Object System.Drawing.SolidBrush($muted)),72,86)

Box 70 155 300 110 '1. 进入阶段窗口`nEEG / 确认 2s / 停止 2s' ([System.Drawing.Color]::FromArgb(226,245,252)) ([System.Drawing.Color]::FromArgb(70,170,210)) $bold
Box 450 155 320 110 '2. 统一清零`nblink=0  left=0  right=0  both=0' ([System.Drawing.Color]::FromArgb(255,242,204)) ([System.Drawing.Color]::FromArgb(234,170,44)) $bold
Box 850 155 350 110 '3. 当前窗口的新 AUX 数据`n只取末尾 180 ms 判断动作状态' ([System.Drawing.Color]::FromArgb(235,247,239)) ([System.Drawing.Color]::FromArgb(71,164,104)) $bold
Box 1280 155 420 110 '4. 上升沿累加`n未激活 → 激活：对应计数 +1`n持续激活：不重复加' ([System.Drawing.Color]::FromArgb(239,235,252)) ([System.Drawing.Color]::FromArgb(126,91,186)) $bold
Arrow 370 210 450 210
Arrow 770 210 850 210
Arrow 1200 210 1280 210

$g.DrawString('同一个窗口内的累加器',$bold,(New-Object System.Drawing.SolidBrush($ink)),72,330)
Box 80 385 285 110 '眨眼计数器`nblink_count' ([System.Drawing.Color]::FromArgb(231,246,254)) ([System.Drawing.Color]::FromArgb(46,154,207))
Box 405 385 285 110 '左咬肌计数器`nleft_jaw_count' ([System.Drawing.Color]::FromArgb(233,249,240)) ([System.Drawing.Color]::FromArgb(51,157,99))
Box 730 385 285 110 '右咬肌计数器`nright_jaw_count' ([System.Drawing.Color]::FromArgb(233,249,240)) ([System.Drawing.Color]::FromArgb(51,157,99))
Box 1055 385 285 110 '双咬肌计数器`nboth_jaw_count' ([System.Drawing.Color]::FromArgb(255,244,231)) ([System.Drawing.Color]::FromArgb(232,135,54))
Box 1380 385 320 110 '当前窗口唯一状态`nlast_window_emg_counts' ([System.Drawing.Color]::FromArgb(243,240,253)) ([System.Drawing.Color]::FromArgb(123,94,181))
Arrow 222 495 1510 550 '所有事件累计到同一份当前窗口计数'

Box 490 585 410 130 '实时显示（当前窗口）`n顶部：当前时间段计数`n实时解码结果 / 状态文本' ([System.Drawing.Color]::FromArgb(226,245,252)) ([System.Drawing.Color]::FromArgb(70,170,210)) $bold
Box 980 585 410 130 '确认算法（当前确认窗口）`n读取同一组计数`n有效咬肌 → 执行；无咬肌 → 取消' ([System.Drawing.Color]::FromArgb(235,247,239)) ([System.Drawing.Color]::FromArgb(71,164,104)) $bold
Arrow 1540 495 695 585
Arrow 1540 495 1185 585

$g.DrawString('窗口边界与历史记录',$bold,(New-Object System.Drawing.SolidBrush($ink)),72,770)
Box 80 825 470 130 '窗口结束`n冻结本窗口快照，只用于历史表 / JSON 日志`n不会反馈给算法' ([System.Drawing.Color]::FromArgb(245,247,250)) ([System.Drawing.Color]::FromArgb(145,160,175)) $font
Box 670 825 470 130 '切换到下一窗口`n四个计数器全部清零`nactive 标志全部清零' ([System.Drawing.Color]::FromArgb(255,242,204)) ([System.Drawing.Color]::FromArgb(234,170,44)) $font
Box 1260 825 440 130 '下一窗口重新累计`n不会读取上一窗口的计数`n不会跨窗口' ([System.Drawing.Color]::FromArgb(235,247,239)) ([System.Drawing.Color]::FromArgb(71,164,104)) $font
Arrow 550 890 670 890
Arrow 1140 890 1260 890

$g.DrawString('取消状态机：确认窗口内无有效咬肌 → 取消 +1；确认成功 → 取消计数清零；连续取消 5 次 → 回静息态。',$font,(New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(150,72,32))),72,1025)
$bmp.Save($out,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Output $out
