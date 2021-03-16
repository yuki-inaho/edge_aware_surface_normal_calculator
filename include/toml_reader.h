
#pragma once
#include "toml.hpp"
#include <iostream>

class TomlReader
{

private:
    toml::value data_;

public:
    TomlReader(std::string filename)
    {
        ReadParameterFile(filename);
    };

    ~TomlReader(){};

    void ReadParameterFile(std::string filename);
    int ReadIntData(std::string table, std::string key);
    bool ReadBoolData(std::string table, std::string key);
    float ReadFloatData(std::string table, std::string key);
    double ReadDoubleData(std::string table, std::string key);
    std::string ReadStringData(std::string table, std::string key);
};