moe-cache v@VERSION@ for Windows (x64), built from commit @COMMIT@. Experimental.

moe-cache is a plugin for llama.cpp's llama-server that runs MoE models bigger than your RAM:
it keeps the experts that are used most in RAM and reads the rest from the SSD. Output is
identical to stock llama.cpp. If your model fits in RAM, stock llama.cpp is faster: the
launcher then runs it stock by itself.

1. What you need
   - llama.cpp, the official release zip whose name ends in -bin-win-vulkan-x64.zip
     (github.com/ggml-org/llama.cpp/releases; use it for AMD, Intel and NVIDIA cards; tested:
     b11669 and b11966), unpacked to C:\llama
     Unpacked somewhere else (for example in Downloads)? Run this once in a Command Prompt,
     then open a new window:  setx LLAMA_SERVER "D:\path\to\llama-server.exe"
   - Python 3.8 or newer (python.org; tick "Add python.exe to PATH"; if typing python opens the
     Microsoft Store, use py instead)
   - a MoE model in GGUF format, on an SSD, with its size plus 10 GB of disk free (most are
     20-120 GB). Not sure which? python bin\moe_cache.py pick  names one and prints the download
     command (it starts with hf: pip install -U huggingface_hub)
   - for an AMD GPU: Adrenalin driver 32.0.32015.7029 or newer is best
     (older drivers crash on Flash-Next checkpoints; the launcher then turns them off)

   Windows may warn "unknown publisher": the kit is not code-signed (open source, built by
   GitHub Actions from the tagged commit). If Windows asked you to confirm anything when you
   unzipped this, open PowerShell and run once (change the folder to where you unzipped):
     Get-ChildItem C:\moe-cache -Recurse | Unblock-File
   If SmartScreen says "Windows protected your PC": More info > Run anyway.
   Check the zip: Get-FileHash moe-cache-v@VERSION@-windows-x64.zip (PowerShell) must equal the
   line in SHA256SUMS-windows on the release page.

2. Start
   Unzip this kit (e.g. to C:\moe-cache) and double-click moe-cache.bat.
   The control page opens at http://127.0.0.1:8070: pick a model, press Start.
   Start-menu entry:   moe-cache.bat --shortcut
   (moe-cache.exe does the same as moe-cache.bat; it is there for winget and scoop.)

   Or from a command prompt in this folder:
     python bin\moe_cache.py check D:\models\model.gguf --llama-server C:\llama\llama-server.exe --run
     python bin\moe_cache.py server D:\models\model.gguf --llama-server C:\llama\llama-server.exe --ram 24 --open
   --ram is the memory in GB the server may use (leave it out to use what is free).

3. Coding agents (Hermes Agent, OpenCode)
     python bin\moe_cache.py server D:\models\model.gguf --llama-server C:\llama\llama-server.exe --agent
     python bin\moe_cache.py connect opencode --write
     python bin\moe_cache.py connect hermes --write
   The control page has Connect buttons for both.
   Or let the agents start the model when they need it (stops after 15 idle minutes):
     python bin\moe_cache.py on-demand D:\models\model.gguf --llama-server C:\llama\llama-server.exe
   Details: docs\AGENTS.md

4. Good to know
   - Keep the Windows page file at "system managed" (or 14 GB or more): the cache itself
     costs no commit (it keeps experts in read-only views of the model file), but
     Windows counts the GPU's memory against RAM + page file.
   - You do not need to choose a plugin DLL: moe-cache.bat picks the right one. Only if check says
     the plugin and your llama.cpp do not match (an old llama.cpp with ggml 0.25), run once in the kit
     folder:  setx MOE_CACHE_LIB "%CD%\ggml-moe-cache-api2.dll"
   - Problems: docs\WINDOWS.md. Report issues at github.com/IbrahimSiddiqui007/moe-cache/issues
     with the output of "python bin\moe_cache.py check MODEL.gguf --run".
   - Share your speed: python bin\moe_cache.py bench-report D:\models\model.gguf --flush-from D:\other-big-files
     (a model bigger than your RAM; 5-20 minutes) writes a report to paste into a "Benchmark report"
     issue. Recipe: docs\SHARE_YOUR_NUMBERS.md. The report holds no prompts, answers or names.

License: see LICENSE.
