#include <iostream>
#include <fstream>
#include <ie_core.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"
#include <thread>
#include "ngraph/runtime/aligned_buffer.hpp"
using namespace ngraph;
using namespace ov;

int main(int args, char *argv[]){
    if (args < 2)
        exit(-1);
    ov::Core core;
    std::string model_path(argv[1]);
    int num_streams = 1;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    core.set_property("CPU", ov::num_streams(1));
    // //core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
    // core.set_property("CPU", ov::inference_num_threads(1));
    // //core.set_property("CPU", ov::enable_profiling(true));
    auto exeNetwork = core.compile_model(model, "CPU");
    int M = 128;
    int K = 384;
    ngraph::runtime::AlignedBuffer alignedA(M*K*4, 64);
    ov::Tensor input_tensor = ov::Tensor(ov::element::f32, ov::Shape{1, 128, 384}, alignedA.get_ptr<float>());
    auto encoder_infer_ = exeNetwork.create_infer_request();
    encoder_infer_.set_tensor("X", input_tensor);
    auto begin = std::chrono::high_resolution_clock::now();
    size_t count = 10;
    std::cout << "Start to Infer" << std::endl;
    for(size_t n = 0; n < count; n++) {
        encoder_infer_.infer();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "Latency " << total_time / count << " us " << std::endl;
}