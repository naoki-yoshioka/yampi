#ifndef YAMPI_REQUEST_ARRAY_HPP
# define YAMPI_REQUEST_ARRAY_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <iterator>
# include <algorithm>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_cv.hpp>
#   include <boost/type_traits/is_same.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <boost/optional.hpp>
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <yampi/status.hpp>
# ifdef BOOST_NO_CXX11_LAMBDAS
#   include <yampi/detail/construct_status.hpp>
#   include <yampi/detail/cast_index.hpp>
# endif // BOOST_NO_CXX11_LAMBDAS
# include <yampi/persistent_request.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_remove_cv boost::remove_cv
#   define YAMPI_is_same boost::is_same
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif


namespace yampi
{
  // TODO: implement iterators
  namespace request_array_detail
  {
    template <typename Request, std::size_t N>
    class request_array_base
    {
     protected:
      MPI_Request data_[N];

     public:
      typedef Request value_type;
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;
      typedef typename Request::reference reference;
      typedef typename Request::const_reference const_reference;
      typedef MPI_Request* pointer;
      typedef MPI_Request const* const_pointer;

      bool operator==(request_array_base const& other) { return std::equal(data_, data_ + N, other.data_); }

      reference at(size_type const position)
      {
        if (position >= N)
          throw std::out_of_range("out of range");
        return reference(data_[position]);
      }

      const_reference at(size_type const position) const
      {
        if (position >= N)
          throw std::out_of_range("out of range");
        return const_reference(data_[position]);
      }

      reference operator[](size_type const position)
      { assert(position < N); return reference(data_[position]); }

      const_reference operator[](size_type const position) const
      { assert(position < N); return const_reference(data_[position]); }

      reference front() { return reference(data_[0u]); }
      const_reference front() const { return const_reference(data_[0u]); }
      reference back() { return reference(data_[N - 1u]); }
      const_reference back() const { return const_reference(data_[N - 1u]); }
      pointer data() BOOST_NOEXCEPT_OR_NOTHROW { return data_; }
      const_pointer data() const BOOST_NOEXCEPT_OR_NOTHROW { return data_; }

      BOOST_CONSTEXPR bool empty() const BOOST_NOEXCEPT_OR_NOTHROW { return false; }
      BOOST_CONSTEXPR size_type size() const BOOST_NOEXCEPT_OR_NOTHROW { return N; }
      BOOST_CONSTEXPR size_type max_size() const BOOST_NOEXCEPT_OR_NOTHROW { return N; }

      std::pair< ::yampi::status, size_type > wait_any(::yampi::environment const& environment)
      {
        MPI_Status mpi_status;
        int index;
        int const error_code = MPI_Waitany(N, data_, YAMPI_addressof(index), YAMPI_addressof(mpi_status));

        return error_code == MPI_SUCCESS
          ? index != MPI_UNDEFINED
            ? std::make_pair(::yampi::status(mpi_status), static_cast<size_type>(index))
            : std::make_pair(::yampi::status(mpi_status), N)
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_any", environment);
      }

      size_type wait_any(::yampi::ignore_status_t const, ::yampi::environment const& environment)
      {
        int index;
        int const error_code = MPI_Waitany(N, data_, YAMPI_addressof(index), MPI_STATUS_IGNORE);

        return error_code == MPI_SUCCESS
          ? index != MPI_UNDEFINED
            ? static_cast<size_type>(index)
            : N
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_any", environment);
      }

      boost::optional< std::pair< ::yampi::status, size_type > > test_any(::yampi::environment const& environment)
      {
        MPI_Status mpi_status;
        int index;
        int flag;
        int const error_code = MPI_Testany(N, data_, YAMPI_addressof(index), YAMPI_addressof(flag), YAMPI_addressof(mpi_status));

        return error_code == MPI_SUCCESS
          ? static_cast<bool>(flag)
            ? index != MPI_UNDEFINED
              ? boost::make_optional(std::make_pair(::yampi::status(mpi_status), static_cast<size_type>(index)))
              : boost::make_optional(std::make_pair(::yampi::status(mpi_status), N))
            : boost::none
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_any", environment);
      }

      boost::optional<size_type> test_any(::yampi::ignore_status_t const, ::yampi::environment const& environment)
      {
        int index;
        int flag;
        int const error_code = MPI_Testany(N, data_, YAMPI_addressof(index), YAMPI_addressof(flag), MPI_STATUS_IGNORE);

        return error_code == MPI_SUCCESS
          ? static_cast<bool>(flag)
            ? index != MPI_UNDEFINED
              ? boost::make_optional(static_cast<size_type>(index))
              : boost::make_optional(N)
            : boost::none
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_any", environment);
      }

      template <typename ContiguousIterator>
      ContiguousIterator wait_all(ContiguousIterator const out, ::yampi::environment const& environment)
      {
        static_assert(
          (YAMPI_is_same<
             typename YAMPI_remove_cv<
               typename std::iterator_traits<ContiguousIterator>::value_type>::type,
             ::yampi::status>::value),
          "Value type of ContiguousIterator must be the same to ::yampi::status");

        MPI_Status mpi_statuses[N];
        int const error_code = MPI_Waitall(N, data_, mpi_statuses);

# ifndef BOOST_NO_CXX11_LAMBDAS
        std::transform(
          mpi_statuses, mpi_statuses + N, out,
          [](MPI_Status const& mpi_status) { return ::yampi::status(mpi_status); });
# else // BOOST_NO_CXX11_LAMBDAS
        std::transform(mpi_statuses, mpi_statuses + N, out, ::yampi::detail::construct_status());
# endif // BOOST_NO_CXX11_LAMBDAS

        return error_code == MPI_SUCCESS
          ? out
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_all", environment);
      }

      void wait_all(::yampi::environment const& environment)
      {
        int const error_code = MPI_Waitall(N, data_, MPI_STATUSES_IGNORE);
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_all", environment);
      }

      template <typename ContiguousIterator>
      std::pair<bool, ContiguousIterator> test_all(ContiguousIterator const out, ::yampi::environment const& environment)
      {
        static_assert(
          (YAMPI_is_same<
             typename YAMPI_remove_cv<
               typename std::iterator_traits<ContiguousIterator>::value_type>::type,
             ::yampi::status>::value),
          "Value type of ContiguousIterator must be the same to ::yampi::status");

        MPI_Status mpi_statuses[N];
        int flag;
        int const error_code = MPI_Testall(N, data_, YAMPI_addressof(flag), mpi_statuses);

# ifndef BOOST_NO_CXX11_LAMBDAS
        std::transform(
          mpi_statuses, mpi_statuses + N, out,
          [](MPI_Status const& mpi_status) { return ::yampi::status(mpi_status); });
# else // BOOST_NO_CXX11_LAMBDAS
        std::transform(mpi_statuses, mpi_statuses + N, out, ::yampi::detail::construct_status());
# endif // BOOST_NO_CXX11_LAMBDAS

        return error_code == MPI_SUCCESS
          ? std::make_pair(static_cast<bool>(flag), out)
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_all", environment);
      }

      bool test_all(::yampi::environment const& environment)
      {
        int flag;
        int const error_code = MPI_Testall(N, data_, YAMPI_addressof(flag), MPI_STATUSES_IGNORE);
        return error_code == MPI_SUCCESS
          ? static_cast<bool>(flag)
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_all", environment);
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2>
      std::pair<bool, ContiguousIterator1> wait_some(
        ContiguousIterator1 const out_index, ContiguousIterator2 const out_status, ::yampi::environment const& environment)
      {
        static_assert(
          (YAMPI_is_same<
             typename YAMPI_remove_cv<
               typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
             size_type>::value),
          "Value type of ContiguousIterator1 must be the same to size_type");
        static_assert(
          (YAMPI_is_same<
             typename YAMPI_remove_cv<
               typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
             ::yampi::status>::value),
          "Value type of ContiguousIterator2 must be the same to ::yampi::status");

        int indices[N];
        MPI_Status mpi_statuses[N];
        int num_completed_requests;
        int const error_code = MPI_Waitsome(N, data_, YAMPI_addressof(num_completed_requests), indices, mpi_statuses);

        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_some", environment);

        if (num_completed_requests == MPI_UNDEFINED)
          return std::make_pair(false, out_index);

# ifndef BOOST_NO_CXX11_LAMBDAS
        std::transform(
          indices, indices + num_completed_requests, out_index,
          [](int const index) { return static_cast<size_type>(index); });
        std::transform(
          mpi_statuses, mpi_statuses + num_completed_requests, out_status,
          [](MPI_Status const& mpi_status) { return ::yampi::status(mpi_status); });
# else // BOOST_NO_CXX11_LAMBDAS
        std::transform(indices, indices + num_completed_requests, out_index, ::yampi::detail::cast_index());
        std::transform(mpi_statuses, mpi_statuses + num_completed_requests, out_status, ::yampi::detail::construct_status());
# endif // BOOST_NO_CXX11_LAMBDAS

        return std::make_pair(true, out_index);
      }

      template <typename ContiguousIterator>
      std::pair<bool, ContiguousIterator> wait_some(
        ContiguousIterator const out_index, ::yampi::environment const& environment)
      {
        static_assert(
          (YAMPI_is_same<
             typename YAMPI_remove_cv<
               typename std::iterator_traits<ContiguousIterator>::value_type>::type,
             size_type>::value),
          "Value type of ContiguousIterator must be the same to size_type");

        int indices[N];
        int num_completed_requests;
        int const error_code = MPI_Waitsome(N, data_, YAMPI_addressof(num_completed_requests), indices, MPI_STATUSES_IGNORE);

        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_some", environment);

        if (num_completed_requests == MPI_UNDEFINED)
          return std::make_pair(false, out_index);

# ifndef BOOST_NO_CXX11_LAMBDAS
        std::transform(
          indices, indices + num_completed_requests, out_index,
          [](int const index) { return static_cast<size_type>(index); });
# else // BOOST_NO_CXX11_LAMBDAS
        std::transform(indices, indices + num_completed_requests, out_index, ::yampi::detail::cast_index());
# endif // BOOST_NO_CXX11_LAMBDAS

        return std::make_pair(true, out_index);
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2>
      std::pair<bool, ContiguousIterator1> test_some(
        ContiguousIterator1 const out_index, ContiguousIterator2 const out_status, ::yampi::environment const& environment)
      {
        static_assert(
          (YAMPI_is_same<
             typename YAMPI_remove_cv<
               typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
             size_type>::value),
          "Value type of ContiguousIterator1 must be the same to size_type");
        static_assert(
          (YAMPI_is_same<
             typename YAMPI_remove_cv<
               typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
             ::yampi::status>::value),
          "Value type of ContiguousIterator2 must be the same to ::yampi::status");

        int indices[N];
        MPI_Status mpi_statuses[N];
        int num_completed_requests;
        int const error_code = MPI_Testsome(N, data_, YAMPI_addressof(num_completed_requests), indices, mpi_statuses);

        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_some", environment);

        if (num_completed_requests == MPI_UNDEFINED)
          return std::make_pair(false, out_index);

# ifndef BOOST_NO_CXX11_LAMBDAS
        std::transform(
          indices, indices + num_completed_requests, out_index,
          [](int const index) { return static_cast<size_type>(index); });
        std::transform(
          mpi_statuses, mpi_statuses + num_completed_requests, out_status,
          [](MPI_Status const& mpi_status) { return ::yampi::status(mpi_status); });
# else // BOOST_NO_CXX11_LAMBDAS
        std::transform(indices, indices + num_completed_requests, out_index, ::yampi::detail::cast_index());
        std::transform(mpi_statuses, mpi_statuses + num_completed_requests, out_status, ::yampi::detail::construct_status());
# endif // BOOST_NO_CXX11_LAMBDAS

        return std::make_pair(true, out_index);
      }

      template <typename ContiguousIterator>
      std::pair<bool, ContiguousIterator> test_some(
        ContiguousIterator const out_index, ::yampi::environment const& environment)
      {
        static_assert(
          (YAMPI_is_same<
             typename YAMPI_remove_cv<
               typename std::iterator_traits<ContiguousIterator>::value_type>::type,
             size_type>::value),
          "Value type of ContiguousIterator must be the same to size_type");

        int indices[N];
        int num_completed_requests;
        int const error_code = MPI_Testsome(N, data_, YAMPI_addressof(num_completed_requests), indices, MPI_STATUSES_IGNORE);

        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_some", environment);

        if (num_completed_requests == MPI_UNDEFINED)
          return std::make_pair(false, out_index);

# ifndef BOOST_NO_CXX11_LAMBDAS
        std::transform(
          indices, indices + num_completed_requests, out_index,
          [](int const index) { return static_cast<size_type>(index); });
# else // BOOST_NO_CXX11_LAMBDAS
        std::transform(indices, indices + num_completed_requests, out_index, ::yampi::detail::cast_index());
# endif // BOOST_NO_CXX11_LAMBDAS

        return std::make_pair(true, out_index);
      }
    };

    template <typename Request, std::size_t N>
    inline bool operator!=(::yampi::request_array_detail::request_array_base<Request, N> const& lhs, ::yampi::request_array_detail::request_array_base<Request, N> const& rhs)
    { return not (lhs == rhs); }
  }

  template <typename Request, std::size_t N>
  class request_array
    : public ::yampi::request_array_detail::request_array_base<Request, N>
  { };

  template <std::size_t N>
  class request_array< ::yampi::persistent_request, N >
    : public ::yampi::request_array_detail::request_array_base< ::yampi::persistent_request, N >
  {
   public:
    void start_all(::yampi::environment const& environment)
    {
      int const error_code = MPI_Startall(N, data_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_array::start_all", environment);
    }
  };
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_remove_cv
# undef YAMPI_is_same

#endif

