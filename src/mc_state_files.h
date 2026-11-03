// CLEANUP-002: model file handles, shard paths and direct-read state
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_files_t {
    std::vector<plat::file_t> fds;              // one per GGUF file (split models have several)
    std::vector<std::string> shard_paths;
    std::vector<plat::file_t> fds_buf;          // buffered (no O_DIRECT) handles, opened only if a direct read keeps failing
    std::mutex fds_buf_mu;
    std::vector<std::atomic<bool>> direct_broken;   // per file: direct reads fail on this file system (EINVAL/EOPNOTSUPP): buffered reads from then on (AUD-011)
    std::unordered_map<std::string, int> file_shard;
};
