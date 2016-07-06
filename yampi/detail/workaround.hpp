#ifndef YAMPI_DETAIL_WORKAROUND_HPP
# define YAMPI_DETAIL_WORKAROUND_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   ifndef __FUJITSU
#     include <utility>
#   endif
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/type_traits/remove_reference.hpp>
#   endif

#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     define YAMPI_remove_reference std::remove_reference
#   else
#     define YAMPI_remove_reference boost::remove_reference
#   endif


namespace yampi
{
  namespace detail
  {
    template <typename T>
    BOOST_CONSTEXPR T&& forward(typename YAMPI_remove_reference<T>::type& t) BOOST_NOEXCEPT_OR_NOTHROW
    { return static_cast<T&&>(t); }

    template <typename T>
    BOOST_CONSTEXPR T&& forward(typename YAMPI_remove_reference<T>::type&& t) BOOST_NOEXCEPT_OR_NOTHROW
    { return static_cast<T&&>(t); }
  }
}

#   ifndef __FUJITSU
#     define YAMPI_DETAIL_forward std::forward
#   else
#     define YAMPI_DETAIL_forward yampi::detail::forward
#   endif

#   undef YAMPI_remove_reference
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

#endif

