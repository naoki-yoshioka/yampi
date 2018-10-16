#ifndef YAMPI_MESSAGE_ENVELOPE_HPP
# define YAMPI_MESSAGE_ENVELOPE_HPP

# include <boost/config.hpp>

# include <utility>

# include <mpi.h>

# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/communicator.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>


namespace yampi
{
  class message_envelope
  {
    ::yampi::rank source_;
    ::yampi::rank destination_;
    ::yampi::tag tag_;
    ::yampi::communicator& communicator_;

   public:
    message_envelope(
      ::yampi::rank const source, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator& communicator)
      : source_(source),
        destination_(destination),
        tag_(tag),
        communicator_(communicator)
    { }

    message_envelope(
      ::yampi::rank const source, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const& communicator)
      : source_(source),
        destination_(destination),
        tag_(tag),
        communicator_(const_cast< ::yampi::communicator& >(communicator))
    { }

    message_envelope(
      ::yampi::rank const source, ::yampi::rank const destination,
      ::yampi::communicator& communicator)
      : source_(source),
        destination_(destination),
        tag_(),
        communicator_(communicator)
    { }

    message_envelope(
      ::yampi::rank const source, ::yampi::rank const destination,
      ::yampi::communicator const& communicator)
      : source_(source),
        destination_(destination),
        tag_(),
        communicator_(const_cast< ::yampi::communicator& >(communicator))
    { }

    ::yampi::rank const& source() const { return source_; }
    ::yampi::rank const& destination() const { return destination_; }
    ::yampi::tag const& tag() const { return tag_; }
    ::yampi::communicator const& communicator() const { return communicator_; }

    void swap(message_envelope& other)
      BOOST_NOEXCEPT_IF(
        ::yampi::utility::is_nothrow_swappable< ::yampi::rank >::value
        and ::yampi::utility::is_nothrow_swappable< ::yampi::tag >::value
        and ::yampi::utility::is_nothrow_swappable< ::yampi::communicator& >::value)
    {
      using std::swap;
      swap(source_, other.source_);
      swap(destination_, other.destination_);
      swap(tag_, other.tag_);
      swap(communicator_, other.communicator_);
    }
  };

  inline void swap(::yampi::message_envelope& lhs, ::yampi::message_envelope& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


#endif

