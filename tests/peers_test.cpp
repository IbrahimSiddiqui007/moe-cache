// Fake-peer test of the peer registry rules (src/mc_peers.h): no model, no second server. g++ -std=c++17 -Isrc tests/peers_test.cpp -o /tmp/peers_test && /tmp/peers_test
#include "mc_peers.h"
#include <cstdio>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
static const uint64_t G = 1ull << 30;
static mcpeers::rec R(long pid, uint64_t rss, uint64_t budget, uint64_t share, int active, int hungry, uint64_t now = 1000) {
    mcpeers::rec r;
    r.pid = pid; r.beat_ms = now; r.rss = rss; r.margin = G; r.budget = budget; r.share = share; r.floor_b = 2 * G; r.active = active; r.hungry = hungry;
    return r;
}
int main() {
    // 1. write and read back; a cut-off or foreign file is refused
    { mcpeers::rec a = R(42, 7 * G, 5 * G, 10 * G, 1, 1), b;
      CHECK(mcpeers::parse(mcpeers::format(a), b) && b.pid == 42 && b.rss == 7 * G && b.budget == 5 * G && b.share == 10 * G && b.active == 1 && b.hungry == 1 && b.floor_b == 2 * G);
      CHECK(!mcpeers::parse("moe-cache-peer 1 42 1000 7", b));
      CHECK(!mcpeers::parse("moe-cache-peer 2 42 1000 1 1 1 1 1 1 1", b));
      CHECK(!mcpeers::parse("", b)); }
    // 2. room = cap - own margin - others' memory and margins; stale and own entries do not count
    { mcpeers::rec me = R(1, 8 * G, 6 * G, 14 * G, 1, 1);
      std::vector<mcpeers::rec> o = { R(1, 99 * G, 0, 0, 0, 0), R(2, 9 * G, 7 * G, 14 * G, 0, 0), R(3, 20 * G, 7 * G, 14 * G, 0, 0, 100) };   // 1 = me, 3 = stale
      auto v = mcpeers::decide(me, o, 28 * G, 1000, 500, 1 * G);                // stale after 500 ms: peer 3 (beat 100) is out, peer 2 (beat 1000) is in
      CHECK(v.live == 1 && v.ceiling == 28 * G - G - (9 * G + G) && v.shrink_to == 0); }
    // 3. a busy hungry server and a quiet one: the quiet one steps down 1 GiB at a time, never below its floor
    { mcpeers::rec quiet = R(2, 12 * G, 10 * G, 14 * G, 0, 0), busy = R(1, 12 * G, 12 * G, 14 * G, 1, 1);
      auto v = mcpeers::decide(quiet, { busy }, 28 * G, 1000, 5000, 1 * G);
      CHECK(v.shrink_to == 9 * G);
      quiet.budget = 2 * G + G / 2;
      v = mcpeers::decide(quiet, { busy }, 28 * G, 1000, 5000, 1 * G);
      CHECK(v.shrink_to == 2 * G);
      quiet.budget = 2 * G;
      v = mcpeers::decide(quiet, { busy }, 28 * G, 1000, 5000, 1 * G);
      CHECK(v.shrink_to == 0); }
    // 4. nobody wants more: the quiet one keeps what it has (no pointless drop)
    { mcpeers::rec quiet = R(2, 12 * G, 10 * G, 14 * G, 0, 0), busy = R(1, 12 * G, 12 * G, 14 * G, 1, 0);
      auto v = mcpeers::decide(quiet, { busy }, 28 * G, 1000, 5000, 1 * G);
      CHECK(v.shrink_to == 0); }
    // 5. both busy: the one above its share steps back only while the other is below its share and hungry
    { mcpeers::rec big = R(1, 20 * G, 18 * G, 12 * G, 1, 1), small = R(2, 5 * G, 3 * G, 12 * G, 1, 1);
      auto v = mcpeers::decide(big, { small }, 28 * G, 1000, 5000, 1 * G);
      CHECK(v.shrink_to == 17 * G);
      v = mcpeers::decide(small, { big }, 28 * G, 1000, 5000, 1 * G);
      CHECK(v.shrink_to == 0);                       // the small one is under its share: it does not give anything
      big.budget = 12 * G;
      v = mcpeers::decide(big, { small }, 28 * G, 1000, 5000, 1 * G);
      CHECK(v.shrink_to == 0); }                     // at its share: stays
    // 6. a dead peer (stopped writing) frees its memory at once for the ceiling
    { mcpeers::rec me = R(1, 8 * G, 6 * G, 14 * G, 1, 1, 9000), dead = R(2, 15 * G, 12 * G, 14 * G, 1, 1, 1000);
      auto v = mcpeers::decide(me, { dead }, 28 * G, 9000, 5000, 1 * G);
      CHECK(v.live == 0 && v.ceiling == 27 * G && v.shrink_to == 0); }
    // 7. no room at all gives 0, not a wrapped number
    { mcpeers::rec me = R(1, 8 * G, 6 * G, 14 * G, 1, 1);
      auto v = mcpeers::decide(me, { R(2, 40 * G, 7 * G, 14 * G, 0, 0) }, 28 * G, 1000, 5000, 1 * G);
      CHECK(v.ceiling == 0); }
#ifndef _WIN32
    // 8. AUD-020: the registry folder is made owner-only, records are 0600, no symlink is followed, a link or foreign file is not "owned by me"
    { char tmpl[] = "/tmp/peers_test_XXXXXX";
      const std::string d = mkdtemp(tmpl);
      chmod(d.c_str(), 0777);
      CHECK(mcpeers::secure_dir(d).empty());
      struct stat st;
      CHECK(stat(d.c_str(), &st) == 0 && (st.st_mode & 077) == 0);
      CHECK(!mcpeers::secure_dir(d + "/nope").empty());
      CHECK(mcpeers::write_private(d + "/1.tmp", "hello"));
      CHECK(stat((d + "/1.tmp").c_str(), &st) == 0 && (st.st_mode & 0777) == 0600 && st.st_size == 5);
      CHECK(mcpeers::owned_by_me(d + "/1.tmp"));
      CHECK(symlink("/etc/passwd", (d + "/2.tmp").c_str()) == 0);
      CHECK(!mcpeers::write_private(d + "/2.tmp", "x"));           // O_NOFOLLOW
      CHECK(!mcpeers::owned_by_me(d + "/2.tmp"));                  // a link is not a record
      CHECK(!mcpeers::owned_by_me(d + "/missing"));
      unlink((d + "/1.tmp").c_str()); unlink((d + "/2.tmp").c_str()); rmdir(d.c_str()); }
#endif
    if (fails) { fprintf(stderr, "%d failed\n", fails); return 1; }
    printf("peers_test: ok\n");
    return 0;
}
