#ifndef YAMPI_ADDRESS_HPP
# define YAMPI_ADDRESS_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>


namespace yampi
{
  class address
  {
    MPI_Aint mpi_address_;

   public:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    address() = default;
    address(address const&) = default;
    address& operator=(address const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    address(address&&) = default;
    address& operator=(address&&) = default;
#   endif
    ~address() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    explicit address(MPI_Aint const mpi_address) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_address_(mpi_address)
    { }

    BOOST_CONSTEXPR bool operator==(address const other) const
    { return mpi_address_ == other.mpi_address_; }

    BOOST_CONSTEXPR bool operator<(address const other) const
    { return mpi_address_ < other.mpi_address_; }

    address operator+(address const lhs, address const rhs) const
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return address(MPI_Aint_add(lhs.mpi_address_, rhs.mpi_address_));
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      MPI_Aint result_mpi_address;
      MPI_Get_address(static_cast<char*>(lhs.mpi_address_) + rhs.mpi_address_, &result_mpi_address);
      return address(result_mpi_address);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    }

    address operator-(address const lhs, address const rhs) const
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return address(MPI_Aint_diff(lhs.mpi_address_, rhs.mpi_address_));
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      MPI_Aint result_mpi_address;
      MPI_Get_address(
        static_cast<char*>(lhs.mpi_address_) - static_cast<char*>(rhs.mpi_address_),
        &result_mpi_address);
      return address(result_mpi_address);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    }

    MPI_Aint const& mpi_address() const { return mpi_address_; }

    void swap(address& other)
      BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_address_, other.mpi_address_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::address const lhs, ::yampi::address const rhs)
  { return not (lhs == rhs); }

  inline BOOST_CONSTEXPR bool operator>=(::yampi::address const lhs, ::yampi::address const rhs)
  { return not (lhs < rhs); }

  inline BOOST_CONSTEXPR bool operator>(::yampi::address const lhs, ::yampi::address const rhs)
  { return rhs < lhs; }

  inline BOOST_CONSTEXPR bool operator<=(::yampi::address const lhs, ::yampi::address const rhs)
  { return not (rhs < lhs); }

  inline ::yampi::address& operator+=(::yampi::address& self, ::yampi::address const other)
  { self = self + other; return self; }

  inline ::yampi::address& operator-=(::yampi::address& self, ::yampi::address const other)
  { self = self - other; return self; }

  inline ::yampi::address& operator++(::yampi::address& self, ::yampi::address const other)
  { self += yampi::address(static_cast<MPI_Aint>(1)); return self; }

  inline ::yampi::address& operator--(::yampi::address& self, ::yampi::address const other)
  { self -= yampi::address(static_cast<MPI_Aint>(1)); return self; }

  inline void swap(::yampi::address& lhs, ::yampi::address& rhs)
    BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable< ::yampi::address >::value)
  { lhs.swap(rhs); }
}

#endif
