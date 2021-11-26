#ifndef YAMPI_MESSAGE_ENVELOPE_HPP
# define YAMPI_MESSAGE_ENVELOPE_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_constructor.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/communicator.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_default_constructible std::is_nothrow_default_constructible
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_default_constructible boost::has_nothrow_default_constructor
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
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
      ::yampi::rank const& source, ::yampi::rank const& destination,
      ::yampi::tag const& tag, ::yampi::communicator const& communicator)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_copy_constructible< ::yampi::rank >::value
        and YAMPI_is_nothrow_copy_constructible< ::yampi::tag >::value)
      : source_(source),
        destination_(destination),
        tag_(tag),
        communicator_ptr_(YAMPI_addressof(communicator))
    { }

    message_envelope(
      ::yampi::rank const& source, ::yampi::rank const& destination,
      ::yampi::communicator const& communicator)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_copy_constructible< ::yampi::rank >::value
        and YAMPI_is_nothrow_default_constructible< ::yampi::tag >::value)
      : source_(source),
        destination_(destination),
        tag_(),
        communicator_ptr_(YAMPI_addressof(communicator))
    { }

    ::yampi::rank const& source() const BOOST_NOEXCEPT_OR_NOTHROW { return source_; }
    ::yampi::rank const& destination() const BOOST_NOEXCEPT_OR_NOTHROW { return destination_; }
    ::yampi::tag const& tag() const BOOST_NOEXCEPT_OR_NOTHROW { return tag_; }
    ::yampi::communicator const& communicator() const BOOST_NOEXCEPT_OR_NOTHROW { return *communicator_ptr_; }

    void swap(message_envelope& other) BOOST_NOEXCEPT_OR_NOTHROW
    {
      using std::swap;
      swap(source_, other.source_);
      swap(destination_, other.destination_);
      swap(tag_, other.tag_);
      swap(communicator_ptr_, other.communicator_ptr_);
    }
  };

  inline void swap(::yampi::message_envelope& lhs, ::yampi::message_envelope& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_default_constructible
# undef YAMPI_is_nothrow_copy_constructible

#endif

