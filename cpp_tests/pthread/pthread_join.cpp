#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>

// 用于控制线程运行的原子变量
std::atomic<bool> running(true);

// 用于同步输出的互斥锁
std::mutex print_mutex;
int num = 0;
int wait_max = 0;

using namespace std;
void thread_function(int p_id) {
    using namespace std::chrono;

    // 记录上次执行的时间点
    auto last_time = steady_clock::now();

    while (running) {
        // 获取当前时间点
        auto current_time = steady_clock::now();

        // 计算时间间隔
        auto duration = duration_cast<milliseconds>(current_time - last_time).count();

        // 更新上次执行的时间点
        last_time = current_time;

        // 打印时间间隔
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << p_id <<"\tTime interval: " << duration << " ms" << std::endl;

        // 模拟线程工作
        std::this_thread::sleep_for(milliseconds(rand()%wait_max));
    }
}

int main(int argc, char* argv[]) {
    if(argc != 3)
    {
        cout << "Pls enter seconds, per thread wait max milliseconds" << endl;
    }
    num = atoi(argv[1]);
    wait_max = atoi(argv[2]);
    std::thread t0(thread_function,0);
    std::thread t1(thread_function,1);
    std::thread t2(thread_function,2);
    std::thread t3(thread_function,3);

    // 让线程运行一段时间后停止
    std::this_thread::sleep_for(chrono::seconds(num));

    // 停止线程
    running = false;

    // 等待线程结束
    t0.join();
    t1.join();
    t2.join();
    t3.join();

    return 0;
}