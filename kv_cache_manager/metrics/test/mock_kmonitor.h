#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/core/MutableMetric.h"

namespace kmonitor {
// 注意：由于 kmonitor 类的复杂性，我们在这里使用简化的 mock
// 在实际测试中，可能需要直接测试与真实 kmonitor 的集成
} // namespace kmonitor