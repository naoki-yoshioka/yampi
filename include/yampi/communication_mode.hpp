#ifndef YAMPI_COMMUNICATION_MODE_HPP
# define YAMPI_COMMUNICATION_MODE_HPP

# include <boost/config.hpp>


namespace yampi
{
  namespace mode
  {
    struct standard_communication { };
    struct buffered_communication { };
    struct synchronous_communication { };
    struct ready_communication { };

    BOOST_STATIC_CONSTEXPR ::yampi::mode::standard_communication standard;
    BOOST_STATIC_CONSTEXPR ::yampi::mode::buffered_communication buffered;
    BOOST_STATIC_CONSTEXPR ::yampi::mode::synchronous_communication synchronous;
    BOOST_STATIC_CONSTEXPR ::yampi::mode::ready_communication ready;
  }
}


#endif

