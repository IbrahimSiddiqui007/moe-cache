#!/bin/bash
# The model-free test set, in ONE place (AUD-087). quick.yml calls this; so does a contributor before a commit:
#   tools/run_quick_tests.sh            compile check + every test below (needs python3 and bash; node for ui_test)
#   tools/run_quick_tests.sh --list     print the test names
# A new tests/*_test.py must be added to PY_TESTS (or to NOT_QUICK with a reason): the script fails on a test file it does not know,
# so a test cannot be written and then never run.
# The C++ checks (plugin syntax with both header sets, most tests/*.cpp) stay in quick.yml's "Plugin syntax" step: they need g++ and ggml headers only.
# CPP_TESTS below are standalone programs (no ggml) that this script compiles and runs itself, so a contributor runs them before a commit.
cd "$(dirname "$0")/.." || exit 1

PY_TESTS="ui_test planner_smi_test lan_test flip_key_test guard_wiring_test security_test bench_report_test bench_record_test on_demand_test warm_test awake_test doctor_test desk_margin_test oom_retry_test
 diffusion_test picker_test gguf_hostile_test pack_test launch_plan_test room_test model_id_test linux_bundle_test llama_patched_bundle_test heavy_background_test bench_orphan_test
 budget_test concurrent_test cpu_gpu_split_test dense_share_test dense_kv_fit_test loadmode_fit_test dense_oom_test commit_guard_test chat_template_fix_test model_folders_test model_parts_test multi_gpu_launch_test multigpu_test project_id_test redirect_key_test plan_rule_test
 cgroup_test planner_test model_links_test public_names_test claims_gate_test conversation_bench_test stage0_records_test update_check_test autostart_test fill_manifests_test connect_agents_test connect_check_test launch_test install_sh_test update_apply_test update_apply_win_test models_catalogue_test wintool_test fast_arg_test launch_state_test sycl_sysman_test agents_state_test sanitize_ci_test fetch_test probe_cache_test fit_cache_test plan_shell_test doctor_arch_test slow_path_test readme_margin_test launcher_small_test cpu_only_test hip_hint_test win_memcap_test workflow_yaml_test stitch_gguf_test download_test agent_guess_test chat_images_test servers_test print_systemd_test undefined_names_test launch_stage_test autotune_test golden_calls_diff tested_commits_test llama_search_test llama_path_test expert_mix_test doc_links_test stage2_arms_test long_prompt_test approx_test token_modes_test monitor_test top_test audit_test hw_win_test top_native_test hard_tasks_test suite_record_test"
SH_TESTS="apply_llama_patches_test release_gate_test release_src_test set_version_test friend_scrub_test win_launcher_test build_api_test"
CPP_TESTS="ctx_size gpu_fallback abort lru_rule names devpick ramguard release_rule cat open_hint pack_rule"       # PREFILL-172 (GPU error -> CPU result, no half-written state), SERVER-012 (abort callback stops the plugin's chunks)
# tests that exist but are not part of the list above, with the reason
NOT_QUICK="require_node_test"      # run below only when CI or REQUIRE_NODE is set: it fails on a machine without node (ui_test would skip its JS checks silently)

if [ "$1" = "--list" ]; then echo $PY_TESTS $SH_TESTS $CPP_TESTS; exit 0; fi

fail=0
for f in tests/*_test.py tests/*_test.sh; do
  n="$(basename "${f%.*}")"
  case " $(echo $PY_TESTS) $SH_TESTS $NOT_QUICK " in *" $n "*) ;; *) echo "FAIL: $f is in neither PY_TESTS/SH_TESTS nor NOT_QUICK in tools/run_quick_tests.sh"; fail=1;; esac
done
[ "$fail" = 0 ] || exit 1

echo "::group::py_compile"
python3 -m py_compile bin/*.py tools/*.py tests/*.py planner/*.py || exit 1
echo "::endgroup::"

if [ -n "$CI$REQUIRE_NODE" ]; then
  export REQUIRE_NODE=1
  echo "::group::require_node_test"; python3 tests/require_node_test.py || exit 1; echo "::endgroup::"
fi

for t in $PY_TESTS; do
  [ -f "tests/$t.py" ] || { echo "FAIL: tests/$t.py is listed in tools/run_quick_tests.sh but does not exist"; exit 1; }
  echo "::group::$t"; python3 "tests/$t.py" || { echo "FAIL: $t"; exit 1; }; echo "::endgroup::"
done
for t in $SH_TESTS; do
  echo "::group::$t"; bash "tests/$t.sh" || { echo "FAIL: $t"; exit 1; }; echo "::endgroup::"
done
if command -v g++ >/dev/null 2>&1; then
  for t in $CPP_TESTS; do
    echo "::group::${t}_test (C++)"
    g++ -std=c++17 -Wall -Wextra -Werror -Isrc "tests/${t}_test.cpp" -o "${TMPDIR:-/tmp}/moe_${t}_test.$$" -lpthread && "${TMPDIR:-/tmp}/moe_${t}_test.$$" || { rm -f "${TMPDIR:-/tmp}/moe_${t}_test.$$"; echo "FAIL: ${t}_test"; exit 1; }
    rm -f "${TMPDIR:-/tmp}/moe_${t}_test.$$"
    echo "::endgroup::"
  done
else
  echo "no g++: C++ tests skipped ($CPP_TESTS)"
  [ -z "$CI" ] || { echo "FAIL: CI needs g++"; exit 1; }
fi
# every MOE_CACHE_ name the code reads must be in README.md or docs/CONFIGURATION.md (DOCS-089, strict since the Critic's round 4); fails the run
python3 tools/env_names_check.py --config-only --strict >/dev/null || { echo "FAIL: MOE_CACHE_ names the code reads and CONFIGURATION.md/README.md do not mention:"; python3 tools/env_names_check.py --config-only --missing 2>/dev/null; exit 1; }
echo "quick tests: all passed"
