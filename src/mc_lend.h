// moe-cache plugin part: how much of a lent expert-cache budget comes back at the first decode graph (AUD-021, idea P20). Plain C++, tested by tests/lend_test.cpp.
#pragma once
#include <algorithm>
#include <cstddef>

namespace mclend {

// lent: bytes taken at the long prompt. budget_after_lend: the budget right after that. budget_now: the budget at the restore.
// If tune() lowered the budget meanwhile (the system had less memory than expected), that reduction is part of what the lend already gave back, so
// adding all of `lent` would put the budget above what tune() just decided. A budget that grew meanwhile gets the whole amount (the caller caps it).
inline size_t restore_amount(size_t lent, size_t budget_after_lend, size_t budget_now) {
    const size_t shrunk = budget_after_lend > budget_now ? budget_after_lend - budget_now : 0;
    return lent > shrunk ? lent - shrunk : 0;
}

}   // namespace mclend
