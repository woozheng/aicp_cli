# AICP_CLI
[中文](README_CN.md) | [English](README.md) 
> An LLM runtime CLI driven by the [AICP (Agent Interaction & Communication Protocol)](https://github.com/woozheng/aicp) — the world's smallest demand-execution Agent, with 2000+ lines of code, 8 files, 176KB, cross-platform. Created by [AICP Engine](https://github.com/woozheng/aicp_engine).

## Capability Boundaries

Demand-oriented, from chat, web search, command execution, task handling,

Operate your computer, check hardware, open camera, organize files, download files,

Capabilities are limited only by your network restrictions and hardware limitations.

## Core Differences from Traditional AI, Claude Hermers, and Other CLIs

No need to install any frameworks, no need to download any plugins, no need to install any tools.

Just need Python 3.11+, your API key, and the protocol drives itself.

AICP_CLI has only one goal: to solve your needs. Not just output but directly execute!

If you already have any mature cloud-based agent API address and key, just tell it and it will use them directly, with no memory retention and no leakage.

No need to configure any identity, capabilities, soul, or other redundant settings,

Stateless, memoryless AICP CLI runtime — one conversation, straight to the result, no token burning!

## Installation

```bash
# 1. Clone repository and enter project folder
git clone https://github.com/woozheng/aicp_cli.git
cd aicp_cli

# 2. Edit config file to set API key and model name
# For Linux / Mac / WSL
nano aicp.yaml
# For Windows
notepad aicp.yaml
# Fill in api_key and model inside aicp.yaml

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the CLI tool
# Windows (use built-in aicp.bat script)
aicp

# Linux / Mac / WSL
python3 aicp.py

```

Supplementary Notes
1、Windows comes with aicp.bat for one-click launch.
2、Use python3 aicp.py to start on Linux, macOS or WSL environments.
3、All version constraints in requirements.txt use standard == syntax for cross-platform compatibility.
4、api_key and model are mandatory fields in aicp.yaml, otherwise the tool cannot send requests normally.
5、Python runtime is required before executing above commands.

## Use Cases
Here's what you can ask it to do:

```text
1. Help me open the camera and take a photo.
2. Help me analyze the c:\project directory and generate a README.md.
3. Help me generate a research report MD for stock 'xxxxx', with the latest data.
4. Help me check my computer processes, identify suspicious virus processes, and output MD.
5. Generate a Gomoku game for me and open it so I can play.
6. Write a Go service for me, listening on port 8888. I already have the environment, just compile and run it.
7. Directly download Docker for me and install it.
8. Search Douban for the top ten movies, with detailed introductions, and generate an MD file.
9. Search GitHub for the hottest AI projects and generate an analysis file.
10. Analyze the current mainstream projects in AI code agents, current status, monetization capabilities, development trends, and future vision. Output a PPT.
11. I want train photos — go download a few from the internet for me.
12. Add watermarks to all images in c:\xxx (or remove watermarks).
13. Generate a standard elementary-to-middle-school Chinese language test paper, at a key middle school level, without answers. Make it difficult!!
14. Read the xxxxxx test paper.md file and automatically generate the answers!
15. Draw a big watermelon for me, then open it for me to see. [Without remote API calls, it will actually hand-draw one]
16. Generate a rhythmic music file for me, 10 seconds long, then play it. [Let's see its aesthetic sense]
17. Generate a voice file saying "I love you" in Chinese, repeat it ten times, and play it.
....
```
Chatting??? What's the point? The clearer your needs, the richer the results. It all depends on what you want to do.
Recommended Models
For coding and programming tasks, prioritize: DeepSeek, Doubao Code, Qwen Coder

For general-purpose, entertainment, or knowledge-based tasks — not recommended.

## 🤝 Contributions
Welcome to submit Issues / PRs. You can add memory to the CLI, or add the built-in -r command to connect with AICP-engine, enabling remote mutual invocation between CLIs.