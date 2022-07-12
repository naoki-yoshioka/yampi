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
  yampi::environment env(argc, argv);

  std::cout << "I am " << yampi::world_communicator.rank(env) << " of " << yampi::world_communicator.size(env) << std::endl;

  return EXIT_SUCCESS;
}
```

