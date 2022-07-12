#ifndef YAMPI_ADDRESS_HPP
# define YAMPI_ADDRESS_HPP

# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <yampi/byte_displacement.hpp>

# include <mpi.h>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class address
  {
    MPI_Aint mpi_address_;

   public:
    constexpr address() : mpi_address_{} { }

    address(address const&) = default;
    address& operator=(address const&) = default;
    address(address&&) = default;
    address& operator=(address&&) = default;
    ~address() noexcept = default;

    explicit constexpr address(MPI_Aint const& mpi_address)
      noexcept(std::is_nothrow_copy_constructible<MPI_Aint>::value)
      : mpi_address_{mpi_address}
    { }

    constexpr bool operator==(address const& other) const
      noexcept(noexcept(mpi_address_ == other.mpi_address_))
    { return mpi_address_ == other.mpi_address_; }

    constexpr bool operator<(address const& other) const
      noexcept(noexcept(mpi_address_ < other.mpi_address_))
    { return mpi_address_ < other.mpi_address_; }

# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    address& operator++() noexcept
    {
      mpi_address_ = MPI_Aint_add(mpi_address_, static_cast<MPI_Aint>(1));
      return *this;
    }
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    address& operator++()
    {
      MPI_Get_address(static_cast<char*>(mpi_address_) + static_cast<MPI_Aint>(1), &mpi_address_);
      return *this;
    }
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)

# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    address& operator--() noexcept
    {
      mpi_address_ = MPI_Aint_diff(mpi_address_, static_cast<MPI_Aint>(1));
      return *this;
    }
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    address& operator--()
    {
      MPI_Get_address(static_cast<char*>(mpi_address_) - static_cast<MPI_Aint>(1), &mpi_address_);
      return *this;
    }
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)

# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    address& operator+=(::yampi::byte_displacement const& displacement) noexcept
    {
      mpi_address_ = MPI_Aint_add(mpi_address_, displacement.mpi_byte_displacement());
      return *this;
    }
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    address& operator+=(::yampi::byte_displacement const& displacement)
    {
      MPI_Get_address(static_cast<char*>(mpi_address_) + displacement.mpi_byte_displacement(), &mpi_address_);
      return *this;
    }
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)

# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    address& operator-=(::yampi::byte_displacement const& displacement) noexcept
    {
      mpi_address_ = MPI_Aint_diff(mpi_address_, displacement.mpi_byte_displacement());
      return *this;
    }
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    address& operator-=(::yampi::byte_displacement const& displacement)
    {
      MPI_Get_address(static_cast<char*>(mpi_address_) - displacement.mpi_byte_displacement(), &mpi_address_);
      return *this;
    }
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)

    constexpr MPI_Aint const& mpi_address() const noexcept { return mpi_address_; }

    void swap(address& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_address_, other.mpi_address_);
    }
  };

  inline constexpr bool operator!=(::yampi::address const& lhs, ::yampi::address const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline constexpr bool operator>=(::yampi::address const& lhs, ::yampi::address const& rhs) noexcept(noexcept(lhs < rhs))
  { return not (lhs < rhs); }

  inline constexpr bool operator>(::yampi::address const& lhs, ::yampi::address const& rhs) noexcept(noexcept(lhs < rhs))
  { return rhs < lhs; }

  inline constexpr bool operator<=(::yampi::address const& lhs, ::yampi::address const& rhs) noexcept(noexcept(lhs < rhs))
  { return not (rhs < lhs); }

  inline ::yampi::address operator++(::yampi::address& self, int)
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    noexcept
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
  { ::yampi::address result = self; ++self; return result; }

  inline ::yampi::address operator--(::yampi::address& self, int)
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    noexcept
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
  { ::yampi::address result = self; --self; return result; }

  inline ::yampi::address operator+(::yampi::address address, ::yampi::byte_displacement const& displacement)
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    noexcept
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
  { address += displacement; return address; }

  inline ::yampi::address operator+(::yampi::byte_displacement const& displacement, ::yampi::address const& address)
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    noexcept
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
  { return address + displacement; }

  inline ::yampi::address operator-(::yampi::address address, ::yampi::byte_displacement const& displacement)
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    noexcept
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
  { address -= displacement; return address; }

  inline ::yampi::byte_displacement operator-(::yampi::address const& lhs, ::yampi::address const& rhs)
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
    noexcept
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
  { return ::yampi::byte_displacement(lhs.mpi_address() - rhs.mpi_address()); }

  inline void swap(::yampi::address& lhs, ::yampi::address& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif
