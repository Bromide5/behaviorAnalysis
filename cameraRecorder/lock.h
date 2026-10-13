#ifndef LOCK_H
#define LOCK_H

#include <shared_mutex>

// 跨文件共享的读写锁：保护 PylonCameraData::opencvImg。
// 写方（取帧线程）用 lock()（独占），读方（录像/显示/推理）用 lock_shared()（共享）。
extern std::shared_mutex g_shared_mutex;

#endif // LOCK_H
