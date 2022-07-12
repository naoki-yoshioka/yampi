#ifndef YAMPI_WINDOW_BASE_HPP
# define YAMPI_WINDOW_BASE_HPP

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/byte_displacement.hpp>
# include <yampi/group.hpp>


namespace yampi
{
# if MPI_VERSION >= 3
  enum class flavor
    : int
  {
    window = MPI_WIN_FLAVOR_CREATE, array = MPI_WIN_FLAVOR_ALLOCATE,
    dynamic = MPI_WIN_FLAVOR_DYNAMIC, array_shared = MPI_WIN_FLAVOR_SHARED
  };

  enum class memory_model
    : int
  { separate = MPI_WIN_SEPARATE, unified = MPI_WIN_UNIFIED };
# endif // MPI_VERSION >= 3

  template <typename Derived>
  class window_base
  {
   public:
    bool is_null() const noexcept { return derived().do_is_null(); }

    MPI_Win const& mpi_win() const noexcept { return derived().do_mpi_win(); }

    template <typename T>
    T* base_ptr(::yampi::environment const& environment) const { return derived().template do_base_ptr<T>(environment); }
    ::yampi::byte_displacement size_bytes(::yampi::environment const& environment) const { return derived().do_size_bytes(environment); }
    int displacement_unit(::yampi::environment const& environment) const { return derived().do_displacement_unit(environment); }
# if MPI_VERSION >= 3
    ::yampi::flavor flavor(::yampi::environment const& environment) const { return derived().do_flavor(environment); }
    ::yampi::memory_model memory_model(::yampi::environment const& environment) const { return derived().do_memory_model(environment); }
# endif // MPI_VERSION >= 3

    void group(::yampi::group& group, ::yampi::environment const& environment) const { derived().do_group(group, environment); }

   protected:
    Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    Derived const& derived() const noexcept { return static_cast<Derived const&>(*this); }
  };
}


#endif

