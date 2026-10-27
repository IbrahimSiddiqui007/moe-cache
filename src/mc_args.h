// moe-cache plugin part: the llama-server settings the plugin has to mirror to stay identical to stock (repacking, host buffer, devices).
// Plain C++ without ggml, so tests/args_test.cpp can check it on its own. llama.cpp's common argument parser applies the environment first and
// then the command line, where the last occurrence wins; flags take their value as the next argument ("--device Vulkan1", no "=" form).
#pragma once
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

namespace mcargs {

// llama.cpp's common_arg_utils::is_truthy / is_falsey
inline bool truthy(const char * v) {
    return v && (!strcmp(v, "on") || !strcmp(v, "enabled") || !strcmp(v, "true") || !strcmp(v, "1"));
}
inline bool falsey(const char * v) {
    return v && (!strcmp(v, "off") || !strcmp(v, "disabled") || !strcmp(v, "false") || !strcmp(v, "0"));
}

inline bool is_one_of(const std::string & a, std::initializer_list<const char *> names) {
    for (const char * n : names) {
        if (a == n) {
            return true;
        }
    }
    return false;
}

struct server_args {
    bool repack = true;                 // --repack / -nr --no-repack (LLAMA_ARG_REPACK)
    bool no_host = false;               // --no-host (LLAMA_ARG_NO_HOST): no GPU host buffer, so CPU weights may go to the repack buffer type
    bool device_none = false;           // --device none (LLAMA_ARG_DEVICE=none): llama.cpp offloads to no GPU
    std::vector<std::string> devices;   // --device A,B: the devices llama.cpp uses (empty: its default choice)
};

inline void set_devices(server_args & r, const std::string & v) {
    r.devices.clear();
    r.device_none = v == "none";
    if (r.device_none) {
        return;
    }
    for (size_t b = 0; b <= v.size();) {
        size_t e = v.find(',', b);
        if (e == std::string::npos) {
            e = v.size();
        }
        if (e > b) {
            r.devices.push_back(v.substr(b, e - b));
        }
        b = e + 1;
    }
}

// args: the process command line (args[0] is the program); env_*: the values of LLAMA_ARG_REPACK, LLAMA_ARG_NO_HOST, LLAMA_ARG_DEVICE (or nullptr)
inline server_args parse_server_args(const std::vector<std::string> & args, const char * env_repack, const char * env_no_host, const char * env_device) {
    server_args r;
    if (falsey(env_repack)) {
        r.repack = false;
    } else if (truthy(env_repack)) {
        r.repack = true;
    }
    r.no_host = truthy(env_no_host);
    if (env_device && *env_device) {
        set_devices(r, env_device);
    }
    for (size_t i = 1; i < args.size(); i++) {
        const std::string & a = args[i];
        if (is_one_of(a, { "-nr", "--no-repack" })) {
            r.repack = false;
        } else if (a == "--repack") {
            r.repack = true;
        } else if (a == "--no-host") {
            r.no_host = true;
        } else if (is_one_of(a, { "-dev", "--device" }) && i + 1 < args.size()) {
            set_devices(r, args[++i]);
        }
    }
    return r;
}

}  // namespace mcargs
