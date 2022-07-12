#ifndef YAMPI_MESSAGE_ENVELOPE_HPP
# define YAMPI_MESSAGE_ENVELOPE_HPP

# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/communicator.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class message_envelope
  {
    ::yampi::rank source_;
    ::yampi::rank destination_;
    ::yampi::tag tag_;
    ::yampi::communicator const* communicator_ptr_;

   public:
    message_envelope(
      ::yampi::rank const source, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const& communicator)
      noexcept(
        std::is_nothrow_copy_constructible< ::yampi::rank >::value
        and std::is_nothrow_copy_constructible< ::yampi::tag >::value)
      : source_{source},
        destination_{destination},
        tag_{tag},
        communicator_ptr_{std::addressof(communicator)}
    { }

    message_envelope(
      ::yampi::rank const source, ::yampi::rank const destination,
      ::yampi::communicator const& communicator)
      noexcept(
        std::is_nothrow_copy_constructible< ::yampi::rank >::value
        and std::is_nothrow_default_constructible< ::yampi::tag >::value)
      : source_{source},
        destination_{destination},
        tag_{},
        communicator_ptr_{std::addressof(communicator)}
    { }

    ::yampi::rank const& source() const noexcept { return source_; }
    ::yampi::rank const& destination() const noexcept { return destination_; }
    ::yampi::tag const& tag() const noexcept { return tag_; }
    ::yampi::communicator const& communicator() const noexcept { return *communicator_ptr_; }

    void swap(message_envelope& other) noexcept
    {
      using std::swap;
      swap(source_, other.source_);
      swap(destination_, other.destination_);
      swap(tag_, other.tag_);
      swap(communicator_ptr_, other.communicator_ptr_);
    }
  };

  inline void swap(::yampi::message_envelope& lhs, ::yampi::message_envelope& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

