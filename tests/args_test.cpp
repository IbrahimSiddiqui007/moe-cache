// Checks src/mc_args.h: the llama-server settings the plugin mirrors (repacking, --no-host, --device), parsed the way llama.cpp's common
// parser does (environment first, then the command line, last occurrence wins).
//   g++ -std=c++17 -Wall -Werror -Isrc tests/args_test.cpp -o /tmp/args_test && /tmp/args_test
#include "mc_args.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using mcargs::parse_server_args;
    using V = std::vector<std::string>;
    {
        const auto r = parse_server_args(V{ "llama-server", "-m", "x.gguf" }, nullptr, nullptr, nullptr);
        CHECK(r.repack && !r.no_host && !r.device_none && r.devices.empty());
    }
    {   // flags and env names
        CHECK(!parse_server_args(V{ "s", "-nr" }, nullptr, nullptr, nullptr).repack);
        CHECK(!parse_server_args(V{ "s", "--no-repack" }, nullptr, nullptr, nullptr).repack);
        CHECK(!parse_server_args(V{ "s" }, "0", nullptr, nullptr).repack);
        CHECK(!parse_server_args(V{ "s" }, "disabled", nullptr, nullptr).repack);
        CHECK(parse_server_args(V{ "s" }, "junk", nullptr, nullptr).repack);       // llama.cpp rejects it; the default stays
        CHECK(parse_server_args(V{ "s", "--no-host" }, nullptr, nullptr, nullptr).no_host);
        CHECK(parse_server_args(V{ "s" }, nullptr, "1", nullptr).no_host);
        CHECK(!parse_server_args(V{ "s" }, nullptr, "0", nullptr).no_host);
    }
    {   // the last occurrence wins; the command line beats the environment
        CHECK(parse_server_args(V{ "s", "--no-repack", "--repack" }, nullptr, nullptr, nullptr).repack);
        CHECK(!parse_server_args(V{ "s", "--no_repack" }, nullptr, nullptr, nullptr).repack);          // llama.cpp turns _ into - in -- options
        CHECK(parse_server_args(V{ "s", "--no_host" }, nullptr, nullptr, nullptr).no_host);
        CHECK(!parse_server_args(V{ "s", "--repack", "-nr" }, nullptr, nullptr, nullptr).repack);
        CHECK(parse_server_args(V{ "s", "--repack" }, "false", nullptr, nullptr).repack);
        CHECK(!parse_server_args(V{ "s", "--device", "Vulkan1" }, nullptr, nullptr, "none").device_none);
    }
    {   // devices
        const auto a = parse_server_args(V{ "s", "-dev", "none" }, nullptr, nullptr, nullptr);
        CHECK(a.device_none && a.devices.empty());
        const auto b = parse_server_args(V{ "s", "--device", "Vulkan1,Vulkan0" }, nullptr, nullptr, nullptr);
        CHECK(!b.device_none && b.devices.size() == 2 && b.devices[0] == "Vulkan1" && b.devices[1] == "Vulkan0");
        const auto c = parse_server_args(V{ "s" }, nullptr, nullptr, "none");
        CHECK(c.device_none);
        const auto d = parse_server_args(V{ "s", "--device" }, nullptr, nullptr, nullptr);      // value missing: ignored
        CHECK(!d.device_none && d.devices.empty());
    }
    {   // CACHE-269 (b): the one-line note about llama-server's own prompt cache is printed only when nothing limits it
        using mcargs::prompt_cache_unlimited_default;
        CHECK(prompt_cache_unlimited_default(V{ "llama-server", "-m", "x.gguf" }, nullptr, nullptr));
        CHECK(!prompt_cache_unlimited_default(V{ "s", "--cache-ram", "2048" }, nullptr, nullptr));
        CHECK(!prompt_cache_unlimited_default(V{ "s", "-cram", "0" }, nullptr, nullptr));
        CHECK(!prompt_cache_unlimited_default(V{ "s", "--cache_ram", "-1" }, nullptr, nullptr));
        CHECK(!prompt_cache_unlimited_default(V{ "s" }, "4096", nullptr));                      // LLAMA_ARG_CACHE_RAM
        CHECK(!prompt_cache_unlimited_default(V{ "s" }, nullptr, "1024"));                      // the launcher (MOE_CACHE_HOST_RESERVE_MIB)
        CHECK(prompt_cache_unlimited_default(V{ "s" }, "", ""));                                // empty env values do not count
        CHECK(prompt_cache_unlimited_default(V{ "--cache-ram", "x", "s" }, nullptr, nullptr));  // args[0] is the program, not an option
    }
    if (fails) {
        fprintf(stderr, "args_test: %d failure(s)\n", fails);
        return 1;
    }
    printf("args_test: all checks passed\n");
    return 0;
}
