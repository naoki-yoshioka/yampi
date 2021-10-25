#ifndef YAMPI_WINDOW_BASE_HPP
# define YAMPI_WINDOW_BASE_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/group.hpp>


namespace yampi
{
  template <typename Derived>
  class window_base
  {
   public:
    bool is_null() const BOOST_NOEXCEPT_OR_NOTHROW { return derived().do_is_null(); }

    MPI_Win const& mpi_win() const BOOST_NOEXCEPT_OR_NOTHROW { return derived().do_mpi_win(); }

    void group(::yampi::group& group, ::yampi::environment const& environment) const { derived().do_group(group, environment); }

   protected:
    Derived& derived() BOOST_NOEXCEPT_OR_NOTHROW { return static_cast<Derived&>(*this); }
    Derived const& derived() const BOOST_NOEXCEPT_OR_NOTHROW { return static_cast<Derived const&>(*this); }
  };
}


#endif

