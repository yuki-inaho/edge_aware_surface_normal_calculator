#include "toml_reader.h"

void TomlReader::ReadParameterFile(std::string filename)
{

    data_ = toml::parse(filename);
}

int TomlReader::ReadIntData(std::string table, std::string key)
{

    const auto tab = toml::get<toml::table>(data_.at(table));
    int read_data = toml::get<int>(tab.at(key));

    return read_data;
}

bool TomlReader::ReadBoolData(std::string table, std::string key)
{

    const auto tab = toml::get<toml::table>(data_.at(table));
    bool read_data = toml::get<bool>(tab.at(key));

    return read_data;
}

float TomlReader::ReadFloatData(std::string table, std::string key)
{

    const auto tab = toml::get<toml::table>(data_.at(table));
    float read_data = toml::get<float>(tab.at(key));

    return read_data;
}

double TomlReader::ReadDoubleData(std::string table, std::string key)
{

    const auto tab = toml::get<toml::table>(data_.at(table));
    double read_data = toml::get<double>(tab.at(key));

    return read_data;
}

std::string TomlReader::ReadStringData(std::string table, std::string key)
{

    const auto tab = toml::get<toml::table>(data_.at(table));
    std::string read_data = toml::get<std::string>(tab.at(key));

    return read_data;
}
