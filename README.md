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
  yampi::world_communicator_t const world_communicator_tag;
  yampi::communicator const comm(world_communicator_tag);

  std::cout << "I am " << comm.rank(env) << " of " << comm.size(env) << std::endl;

  return EXIT_SUCCESS;
}
```

