#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <experimental/optional>
#include <experimental/filesystem>

#include "MemoryMapped.h"

using namespace std;
using std::experimental::optional;
using namespace std::experimental::filesystem;

struct measurement
{
    float z;
    float force;
};

struct dataset
{
    float x;
//     char padding[64-sizeof(float)];
    float y;
//     char padding2[64-sizeof(float)];
    
    vector<measurement> extend;
    vector<measurement> retract;
};

vector<string> split_str(const char *str, char c = ' ')
{
    vector<string> result;

    do
    {
        const char *begin = str;

        while(*str != c && *str)
            str++;

        result.push_back(string(begin, str));
    } while ('\0' != *str++);

    return result;
}

vector<string> split_file_to_lines(const char* str, size_t size, char c='\n')
{
    vector<string> result;
    result.reserve(800); // there will be 700 lines in file

    const char* end = str + size;
    do
    {
        const char *begin = str;

        while(*str != c && *str)
            str++;

        result.push_back(string(begin, str));
    } while (end > str++);

    return result;
}

void parse_lines(const vector<string>& lines, optional<int>& idx_out, dataset& data_out)
{
    bool reading_extend_section = true;
    string str_to_parse;
    for(size_t i=0; i< lines.size(); i++)
    {
        const string& line = lines[i];
        if(line.size() > 1)
        {
            vector<string> tokens = split_str(line.c_str());
            if(tokens[0][0] == '#')
            {
                if(tokens[1] == "index:")
                {
                    int val = stoi(tokens[2]);
                    if(!idx_out)
                    {
                        idx_out = val;
                    }
                    else
                    {
                        if(*idx_out != val)
                        {
                            throw runtime_error("file corrupt, index changed!");
                        }
                    }
                }
                else if(tokens[1] == "xPosition:" || tokens[1] == "yPosition:")
                {
                    float val = std::stof(str_to_parse=tokens[2]);
                    
                    if(tokens[1][0] == 'x')
                    {
                        data_out.x = val;
                    }
                    else
                    {
                        data_out.y = val;
                    }
                }
                else if(tokens[1] == "segment:")
                {
                    if(tokens[2] == "extend")
                    {
                        reading_extend_section = true;
                    }
                    else if(tokens[2] == "retract")
                    {
                        reading_extend_section = false;
                    }
                    else
                    {
                        throw runtime_error("file corrupt, unknown segment: " + tokens[2]);
                    }
                }
            }
            else
            {
                if(idx_out)
                {
                    measurement m;
                    m.z = std::stof(str_to_parse=tokens[0]);
                    m.force = std::stof(str_to_parse=tokens[1]);
                    
                    if(reading_extend_section)
                        data_out.extend.push_back(m);
                    else
                        data_out.retract.push_back(m);
                }
            }
        }
    }
}

vector<string> parse_file_2(const path& path)
{
    ifstream inp(path);
    if(!inp.is_open())
    {
        cerr << "warning: cannot open " << std::quoted(string(path)) << endl;
        return vector<string>();
    }
                    
    vector<string> lines;
    string line;
    while(inp.good())
    {
        getline(inp, line);
        lines.push_back(line);
    }
    
    return lines;
}

vector<string> parse_file(const path& path)
{
    MemoryMapped inp(path, MemoryMapped::MapRange::WholeFile, MemoryMapped::CacheHint::SequentialScan);
    
    string line;
    
    const size_t size = inp.mappedSize();
    vector<string> lines = split_file_to_lines(reinterpret_cast<const char*>(inp.getData()), size);
    
    return lines;
}

int main(int argc, char** argv)
{
    std::map<int, dataset> datasets;
    

    #pragma omp parallel
    {
        #pragma omp single
        for(int i=1; i<argc; i++)
        {
            if(is_directory(argv[i]))
            {
                for (directory_entry dirEntry : directory_iterator(argv[i]))
                {
                    if(is_regular_file(dirEntry.status()))
                    {
    //                         parse_file(dirEntry.path(), datasets);
    //                         cout << std::quoted(string(dirEntry.path())) << "\n";
                        #pragma omp task
                        {
                            vector<string> tmp;
                            tmp = parse_file_2(dirEntry.path());
                            
                            optional<int> idx;
                            dataset data;
                            parse_lines(tmp, idx, data);
                            
                            if(idx)
                            {
                                #pragma omp critical
                                {
                                    datasets[*idx] = data;
                                }
                            }
                            else
                            {
                                cerr << "warning: no index found in file " << std::quoted(string(dirEntry.path())) << endl;
                            }
                        }
                    }
                }
            }
            else
            {
                cerr << "warning: " << quoted(argv[i]) << " is not a directory" << endl;
            }
        }
        
        cudaAnalyse(datasets);
    }
}

