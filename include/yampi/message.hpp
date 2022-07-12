#if MPI_VERSION >= 3
# ifndef YAMPI_MESSAGE_HPP
#   define YAMPI_MESSAGE_HPP

#   include <utility>
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif

#   include <mpi.h>

#   if __cplusplus >= 201703L
#     define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
#   else
#     define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
#   endif


namespace yampi
{
  struct return_message_t { };
# if __cplusplus >= 201703L
  inline constexpr ::yampi::return_message_t return_message{};
# else
  constexpr ::yampi::return_message_t return_message{};
# endif

  class message
  {
    MPI_Message mpi_message_;

   public:
    message()
      noexcept(std::is_nothrow_copy_constructible<MPI_Message>::value)
      : mpi_message_(MPI_MESSAGE_NULL)
    { }

    explicit message(MPI_Message const& mpi_message)
      noexcept(std::is_nothrow_copy_constructible<MPI_Message>::value)
      : mpi_message_(mpi_message)
    { }

    bool is_null() const noexcept(noexcept(mpi_message_ == MPI_MESSAGE_NULL))
    { return mpi_message_ == MPI_MESSAGE_NULL; }

    bool has_null_process() const noexcept(noexcept(mpi_message_ == MPI_MESSAGE_NO_PROC))
    { return mpi_message_ == MPI_MESSAGE_NO_PROC; }

    MPI_Message const& mpi_message() const noexcept { return mpi_message_; }

    void swap(message& other) noexcept(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_message_, other.mpi_message_);
    }
  };

  inline void swap(::yampi::message& lhs, ::yampi::message& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


#   undef YAMPI_is_nothrow_swappable

# endif // YAMPI_MESSAGE_HPP
#endif // MPI_VERSION >= 3

