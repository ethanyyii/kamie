# v90：＝mainline-v89\keepawake_run_v89.ps1，修一個坑：剩下的參數只有一個時被當字串展開。用法：-Log <檔> <exe> <參數…>（不要加 --：-File 模式會把它當成空名字的參數）
# v88：長跑時不讓電腦睡（§8 v87：電源計畫閒置 5 小時會睡）——這個行程掛 SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)，
# 然後在同一個行程裡跑給的指令（跑完自動解除）。螢幕照樣可以關。
#   pwsh -File mainline-v89\keepawake_run_v88.ps1 -Log mainline-v89\logs\x.out -- python replay_v89.py ...
param([Parameter(Mandatory = $true)][string]$Log, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Cmd)
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$flags = [Convert]::ToUInt32('80000001', 16)
$prev = [W.P]::SetThreadExecutionState($flags)
"keepawake set (prev=0x{0:X}) at {1}" -f $prev, (Get-Date -Format s) | Out-File $Log -Encoding utf8
$env:PYTHONIOENCODING = 'utf-8'
$exe = $Cmd[0]
$rest = @($Cmd | Select-Object -Skip 1)          # v90：只剩一個參數時 $rest 會變成字串、splat 只傳出第一個字（實測 "C"）⇒ 一律包成陣列
& $exe @rest *>> $Log
$code = $LASTEXITCODE
[void][W.P]::SetThreadExecutionState([Convert]::ToUInt32('80000000', 16))
"exit $code at {0}" -f (Get-Date -Format s) | Out-File $Log -Append -Encoding utf8
exit $code
