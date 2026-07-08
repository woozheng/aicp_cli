@echo off
setlocal

set SCRIPT_DIR=%~dp0
set AICP_YAML=%SCRIPT_DIR%aicp.yaml

if not exist "%AICP_YAML%" (
    echo ⚠️ 未找到 aicp.yaml，创建默认配置...
    (
        echo models:
        echo   default: deepseek-v4-flash
        echo   providers:
        echo     deepseek:
        echo       api_key: sk-your-key
        echo       base_url: https://api.deepseek.com/v1
        echo       models:
        echo         - id: deepseek-v4-flash
    ) > "%AICP_YAML%"
    echo ✅ 已创建 aicp.yaml，请填入你的 API Key
    pause
    exit /b
)

python "%SCRIPT_DIR%aicp.py" %*