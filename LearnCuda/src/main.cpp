#include <exception>
#include <iostream>
#include <memory>

#include "learn_cuda/application.hpp"

int main(int argc, char* argv[])
{
    try
    {
        std::filesystem::path parentPath = std::filesystem::path(argv[0]).parent_path();
        std::unique_ptr<LearnCuda::Application> app = std::make_unique<LearnCuda::Application>(parentPath);
        if (app) app->Run();
    }
    catch (std::exception& e)
    {
        std::cerr << "\033[31m" << "EXCEPTION: " << e.what() << "\033[0m" << std::endl;
        return -1;
    }

    return 0;
}
