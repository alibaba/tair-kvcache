#pragma once

// Compatibility include for internal users. The comparison API is public so a
// caller can explicitly validate the checksum returned by MatchLocation or
// MatchMeta against its own trusted batch.
#include "kv_cache_manager/client/include/common.h"
