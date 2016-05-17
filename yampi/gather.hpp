#ifndef YAMPI_GATHER_HPP
# define YAMPI_GATHER_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_same.hpp>
# endif
# include <vector>
# include <iterator>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>

# include <mpi.h>

# include <yampi/has_corresponding_mpi_data_type.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/mpi_data_type_of.hpp>
# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_enable_if boost::enable_if
#   define YAMPI_is_same boost::is_same
# endif


namespace yampi
{
  class gather
  {
    ::yampi::rank root_;
    ::yampi::communicator comm_;

   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    BOOST_CONSTEXPR gather() BOOST_NOEXCEPT_OR_NOTHROW
      : root_{0}, comm_{::yampi::world}
    { }

    BOOST_CONSTEXPR gather(::yampi::rank const root) BOOST_NOEXCEPT_OR_NOTHROW
      : root_{root}, comm_{::yampi::world}
    { }

    BOOST_CONSTEXPR gather(::yampi::communicator const comm) BOOST_NOEXCEPT_OR_NOTHROW
      : root_{0}, comm_{comm}
    { }

    BOOST_CONSTEXPR gather(::yampi::rank const root, ::yampi::communicator const comm) BOOST_NOEXCEPT_OR_NOTHROW
      : root_{root}, comm_{comm}
    { }
# else
    BOOST_CONSTEXPR gather() BOOST_NOEXCEPT_OR_NOTHROW
      : root_(0), comm_(::yampi::world)
    { }

    BOOST_CONSTEXPR gather(::yampi::rank const root) BOOST_NOEXCEPT_OR_NOTHROW
      : root_(root), comm_(::yampi::world)
    { }

    BOOST_CONSTEXPR gather(::yampi::communicator const comm) BOOST_NOEXCEPT_OR_NOTHROW
      : root_(0), comm_(comm)
    { }

    BOOST_CONSTEXPR gather(::yampi::rank const root, ::yampi::communicator const comm) BOOST_NOEXCEPT_OR_NOTHROW
      : root_(root), comm_(comm)
    { }
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    gather(gather const&) = default;
    gather& operator=(gather const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    gather(gather&&) = default;
    gather& operator=(gather&&) = default;
#   endif
    ~gather() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    call(
      typename std::iterator_traits<ContiguousIterator>::value_type const& send_value,
      ContiguousIterator const receive_first, ::yampi::environment&) const
    {
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Gather(&send_value, 1, ::yampi::mpi_data_type_of<value_type>::value, &*receive_first, 1, ::yampi::mpi_data_type_of<value_type>::value, root_.mpi_rank(), comm_.mpi_comm());
# else
      int const error_code
        = MPI_Gather(&send_value, 1, ::yampi::mpi_data_type_of<value_type>::value, &*receive_first, 1, ::yampi::mpi_data_type_of<value_type>::value, root_.mpi_rank(), comm_.mpi_comm());
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::gather::call"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call");
# endif
    }

    template <typename Value>
    typename YAMPI_enable_if<::yampi::has_corresponding_mpi_data_type<Value>::value, void>::type
    call(Value const& send_value, ::yampi::environment& env) const
    {
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (comm_.rank(env) == root_)
        throw ::yampi::nonroot_call_on_root_error{"yampi::gather::call"};
# else
      if (comm_.rank(env) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto null = Value{};
#   else
      auto null = Value();
#   endif
# else
      Value null;
# endif

      call(send_value, &null);
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator1>::value_type>::value
        and YAMPI_is_same<typename std::iterator_traits<ContiguousIterator1>::value_type, typename std::iterator_traits<ContiguousIterator2>::value_type>::value,
      void>::type
    call(ContiguousIterator1 const send_first, int const length, ContiguousIterator2 const receive_first, ::yampi::environment&) const
    {
      typedef typename std::iterator_traits<ContiguousIterator1>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Gather(&*send_first, length, ::yampi::mpi_data_type_of<value_type>::value, &*receive_first, length, ::yampi::mpi_data_type_of<value_type>::value, root_.mpi_rank(), comm_.mpi_comm());
# else
      int const error_code
        = MPI_Gather(&*send_first, length, ::yampi::mpi_data_type_of<value_type>::value, &*receive_first, length, ::yampi::mpi_data_type_of<value_type>::value, root_.mpi_rank(), comm_.mpi_comm());
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "mpi::gather"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "mpi::gather");
# endif
    }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    call(ContiguousIterator const send_first, int const length, ::yampi::environment& env) const
    {
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (comm_.rank(env) == root_)
        throw ::yampi::nonroot_call_on_root_error{};
# else
      if (comm_.rank(env) == root_)
        throw ::yampi::nonroot_call_on_root_error();
# endif

      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto null = value_type{};
#   else
      auto null = value_type();
#   endif
# else
      value_type null;
# endif

      call(send_first, length, &null);
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator1>::value_type>::value
        and YAMPI_is_same<typename std::iterator_traits<ContiguousIterator1>::value_type, typename std::iterator_traits<ContiguousIterator2>::value_type>::value,
      void>::type
    call(ContiguousIterator1 const send_first, ContiguousIterator1 const send_last, ContiguousIterator2 const receive_first, ::yampi::environment& env) const
    { call(send_first, send_last-send_first, receive_first, env); }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    call(ContiguousIterator const send_first, ContiguousIterator const last, ::yampi::environment& env) const
    {
      assert(send_last >= send_first);
      call(send_first, send_last-send_first, env);
    }

    template <typename ContiguousRange, typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator>::value
        and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value
        and YAMPI_is_same<typename boost::range_value<ContiguousRange>::type, typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    call(ContiguousRange const& send_values, ContiguousIterator const receive_first, ::yampi::environment& env) const
    { gather(boost::begin(send_values), boost::end(send_values), receive_first, env); }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value
        and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
      void>::type
    call(ContiguousRange const& send_values, ::yampi::environment& env) const
    { gather(boost::begin(send_values), boost::end(send_values), env); }


    template <typename Value>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<Value>::value,
      std::vector<Value> >::type
    to_vector(Value const& send_value, ::yampi::environment& env) const
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto result = std::vector<Value>{};
#   else
      auto result = std::vector<Value>();
#   endif
# else
      std::vector<Value> result;
# endif

      if (comm_.rank(env) == root_)
        result.resize(comm_.size(env));

      call(send_value, result.begin());

      return result;
    }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      std::vector<typename std::iterator_traits<ContiguousIterator>::value_type> >::type
    to_vector(ContiguousIterator const send_first, int const length, ::yampi::environment& env) const
    {
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto result = std::vector<value_type>{};
#   else
      auto result = std::vector<value_type>();
#   endif
# else
      std::vector<value_type> result;
# endif

      if (comm_.rank(env) == root_)
        result.resize(length*comm_.size(env));

      call(send_first, length, result.begin(), env);

      return result;
    }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      std::vector<typename std::iterator_traits<ContiguousIterator>::value_type> >::type
    to_vector(ContiguousIterator const send_first, ContiguousIterator const last, ::yampi::environment& env) const
    {
      assert(send_last >= send_first);
      return to_vector(send_first, send_last-send_first, env);
    }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value
        and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
      std::vector<typename boost::range_value<ContiguousRange>::type> >::type
    to_vector(ContiguousRange const& send_values, ::yampi::environment& env) const
    { return to_vector(boost::begin(send_values), boost::end(send_values), env); }
  };
}


# undef YAMPI_enable_if
# undef YAMPI_is_same

#endif

