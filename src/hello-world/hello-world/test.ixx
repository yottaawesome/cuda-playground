module;
#include <iostream>
export module greet;

export void greet_from_host() {
    std::cout << "Hello from host module!\n";
}