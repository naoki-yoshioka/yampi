#if MPI_VERSION >= 3
# ifndef YAMPI_MESSAGE_HPP
#   define YAMPI_MESSAGE_HPP

#   include <boost/config.hpp>

#   include <utility>
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#     if __cplusplus < 201703L
#       include <boost/type_traits/is_nothrow_swappable.hpp>
#     endif
#   else
#     include <boost/type_traits/has_nothrow_copy.hpp>
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif

#   include <mpi.h>

#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   else
#     define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   endif

#   if __cplusplus >= 201703L
#     define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
#   else
#     define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
#   endif


namespace yampi
{
  struct return_message_t { };

  class message
  {
    MPI_Message mpi_message_;

   public:
    message()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Message>::value)
      : mpi_message_(MPI_MESSAGE_NULL)
    { }

    explicit message(MPI_Message const mpi_message)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Message>::value)
      : mpi_message_(mpi_message)
    { }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_message_ == MPI_MESSAGE_NULL))
    { return mpi_message_ == MPI_MESSAGE_NULL; }

    bool has_null_process() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_message_ == MPI_MESSAGE_NO_PROC))
    { return mpi_message_ == MPI_MESSAGE_NO_PROC; }

    MPI_Message const& mpi_message() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_message_; }

    void swap(message& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_message_, other.mpi_message_);
    }
  };

  inline void swap(::yampi::message& lhs, ::yampi::message& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


#   undef YAMPI_is_nothrow_swappable
#   undef YAMPI_is_nothrow_copy_constructible

# endif // YAMPI_MESSAGE_HPP
#endif // MPI_VERSION >= 3

