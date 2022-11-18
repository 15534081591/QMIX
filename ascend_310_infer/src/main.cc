#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "../inc/utils.h"
#include "include/dataset/execute.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision.h"
#include "include/dataset/vision_ascend.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using mindspore::Serialization;
using mindspore::Model;
using mindspore::Context;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::Graph;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::DataType;

DEFINE_string(model_path, "", "model path");
DEFINE_string(output_path, ".", "output path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_model_path).empty()) {
        std::cout << "Invalid model" << std::endl;
        return 1;
    }

    auto context = std::make_shared<Context>();
    auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310_info);

    Graph graph;
    Status ret = Serialization::Load(FLAGS_model_path, ModelType::kMindIR, &graph);
    if (ret != kSuccess) {
        std::cout << "Load model failed." << std::endl;
        return 1;
    }

    Model model;
    ret = model.Build(GraphCell(graph), context);
    if (ret != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }

    std::vector<MSTensor> modelInputs = model.GetInputs();

    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;

    mindspore::MSTensor agent_inputs =  ReadFileToTensor(std::string(FLAGS_output_path) + "/agent_inputs.bin");
    mindspore::MSTensor h_in =  ReadFileToTensor(std::string(FLAGS_output_path) + "/h_in.bin");
    inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(),
                        modelInputs[0].Shape(), agent_inputs.Data().get(), agent_inputs.DataSize());
    inputs.emplace_back(modelInputs[1].Name(), modelInputs[1].DataType(),
                        modelInputs[1].Shape(), h_in.Data().get(), h_in.DataSize());
    model.Predict(inputs, &outputs);
    WriteResult(std::string(FLAGS_output_path), outputs);

    return 0;
}
