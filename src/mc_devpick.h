// Which ggml devices the plugin may use for the long-prompt GPU block (MULTIGPU-026). Header-only, no ggml: tests/devpick_test.cpp includes it.
// ggml-openvino reports the device chosen by GGML_OPENVINO_DEVICE (for example the NPU of a handheld) as type GPU. The NPU runs static shapes and turns
// weights into constants when they are uploaded, so every block would be a new compile and a new conversion (backends audit, finding 4): never a block target.
#pragma once
#include <cctype>
#include <cstring>

namespace mcdev {

inline bool contains_nocase(const char * hay, const char * needle) {
    if (!hay || !needle || !*needle) {
        return false;
    }
    const size_t nl = strlen(needle);
    for (; *hay; hay++) {
        size_t i = 0;
        while (i < nl && hay[i] && tolower((unsigned char) hay[i]) == tolower((unsigned char) needle[i])) {
            i++;
        }
        if (i == nl) {
            return true;
        }
    }
    return false;
}

// reg_name: ggml_backend_reg_name of the device's registry; desc: ggml_backend_dev_description (either may be null)
inline bool usable_for_gpu_block(const char * reg_name, const char * desc) {
    return !contains_nocase(reg_name, "openvino") && !contains_nocase(desc, "NPU");
}

}  // namespace mcdev
