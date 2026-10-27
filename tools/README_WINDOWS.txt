moe-cache v@VERSION@ for Windows (x64), built from commit @COMMIT@. Experimental.

moe-cache is a plugin for llama.cpp's llama-server that runs MoE models bigger than your RAM:
it keeps the experts that are used most in RAM and reads the rest from the SSD. Output is
identical to stock llama.cpp. If your model fits in RAM, stock llama.cpp is faster: the
launcher then runs it stock by itself.

1. What you need
   - llama.cpp, the official release zip "llama-bNNNNN-bin-win-vulkan-x64.zip"
     (github.com/ggml-org/llama.cpp/releases; tested: b11669), unpacked to C:\llama
   - Python 3.8 or newer (python.org; tick "Add python.exe to PATH")
   - a MoE model in GGUF format, on an SSD
   - for an AMD GPU: Adrenalin driver 32.0.32015.7029 or newer is best
     (older drivers crash on Flash-Next checkpoints; the launcher then turns them off)

2. Start
   Unzip this kit (e.g. to C:\moe-cache) and double-click moe-cache.bat.
   The control page opens at http://127.0.0.1:8070: pick a model, press Start.
   Start-menu entry:   moe-cache.bat --shortcut

   Or from a command prompt in this folder:
     python bin\moe_cache.py check D:\models\model.gguf --llama-server C:\llama\llama-server.exe --run
     python bin\moe_cache.py server D:\models\model.gguf --llama-server C:\llama\llama-server.exe --ram 24 --open
   --ram is the memory in GB the server may use (leave it out to use what is free).

3. Coding agents (Hermes Agent, OpenCode)
     python bin\moe_cache.py server D:\models\model.gguf --llama-server C:\llama\llama-server.exe --agent
     python bin\moe_cache.py connect opencode --write
     python bin\moe_cache.py connect hermes --write
   The control page has Connect buttons for both. Details: docs\AGENTS.md

4. Good to know
   - Keep the Windows page file at "system managed" or 14 GB or more: a small page file
     limits the cache (docs\WINDOWS.md, "where the RAM goes").
   - The plugin DLL is ggml-moe-cache-api3.dll (llama.cpp with ggml 0.26 / 0.27);
     ggml-moe-cache-api2.dll is for older llama.cpp (ggml 0.25). The launcher picks the right one.
   - Problems: docs\WINDOWS.md. Report issues at github.com/IbrahimSiddiqui007/moe-cache/issues
     with the output of "python bin\moe_cache.py check MODEL.gguf --run".

License: see LICENSE.
