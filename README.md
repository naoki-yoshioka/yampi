# yampi

**yampi** (Yet Another MPI library) is an open-source thin-wrapper library of MPI (Message Passing Interface) for C++.

## Examples

```cpp
#include <cstddef>
#include <iostream>

#include <yampi/environment.hpp>
#include <yampi/communicator.hpp>
#include <yampi/rank_io.hpp>

int main(int argc, char* argv[])
{
  yampi::environment env{argc, argv};
  auto comm = yampi::communicator{yampi::tags::world_communicator};

  std::cout << "I am " << comm.rank(env) << " of " << comm.size(env) << std::endl;
  // For C++17 or later
  // std::cout << "I am " << yampi::world_communicator.rank(env) << " of " << yampi::world_communicator.size(env) << std::endl;
}
```

