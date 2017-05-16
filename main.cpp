#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <experimental/optional>
#include <experimental/filesystem>


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
    optional<long> idx;
    float x;
//     char padding[64-sizeof(float)];
    float y;
//     char padding2[64-sizeof(float)];
    
    vector<measurement> extend;
    vector<measurement> retract;
    
    // cuda adress space
    size_t cuBufLen;
    measurement* cuExtend;
    measurement* cuRetract;
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

void parse_lines(const vector<string>& lines, dataset& data_out)
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
                    long val = stol(tokens[2]);
                    if(!data_out.idx)
                    {
                        data_out.idx = val;
                    }
                    else
                    {
                        if(*data_out.idx != val)
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
                else if(tokens[1] == "segment")
                {
                    if(tokens[2].find("extend") != string::npos)
                    {
                        reading_extend_section = true;
                    }
                    else if(tokens[2].find("retract") != string::npos)
                    {
                        reading_extend_section = false;
                    }
                    else
                    {
                        throw runtime_error("file corrupt, unknown segment: '" + tokens[2]+"'");
                    }
                }
            }
            else
            {
                if(data_out.idx)
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

vector<string> parse_file(const path& path)
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



int main(int argc, char** argv)
{
    std::vector<dataset> datasets;
    

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
                        #pragma omp task
                        {
                            vector<string> tmp;
                            tmp = parse_file(dirEntry.path());
                            
                            dataset data;
                            parse_lines(tmp, data);
                            
                            #pragma omp critical
                            {
                                datasets.push_back(data);
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
        
        auto my_comp = [](const measurement &a, const measurement &b){ return a.z < b.z; };
	#pragma omp for schedule(static)
	for(size_t i=0; i<datasets.size(); i++)
	{
	    std::reverse(datasets[i].extend.begin(),datasets[i].extend.end());
	    // data should already be sorted, just to be sure
	    std::sort(datasets[i].extend.begin(),   datasets[i].extend.end(), my_comp); // sort ascending acc. to height z
	    std::sort(datasets[i].retract.begin(),   datasets[i].retract.end(), my_comp); // sort ascending acc. to height z
	}
    }
    
    
//     cudaAnalyse(datasets);
}

