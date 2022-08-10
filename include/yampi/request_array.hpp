#ifndef YAMPI_REQUEST_ARRAY_HPP
# define YAMPI_REQUEST_ARRAY_HPP

# include <cstddef>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <boost/optional.hpp>

# include <yampi/status.hpp>
# include <yampi/persistent_request.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  namespace request_array_detail
  {
    template <typename Request>
    class request_array_const_iterator;

    template <typename Request>
    class request_array_iterator
    {
      friend class request_array_const_iterator<Request>;
      MPI_Request* mpi_request_ptr_;

     public:
      using value_type = Request;
      using pointer = MPI_Request*;
      using reference = typename Request::reference;
      using difference_type = std::ptrdiff_t;
      using iterator_category = std::random_access_iterator_tag;

      constexpr request_array_iterator() noexcept = default;
      constexpr request_array_iterator(MPI_Request& mpi_request) noexcept : mpi_request_ptr_{std::addressof(mpi_request)} { }

      constexpr bool operator==(request_array_iterator const& other) const noexcept { return mpi_request_ptr_ == other.mpi_request_ptr_; }
      constexpr bool operator<(request_array_iterator const& other) const noexcept { return mpi_request_ptr_ < other.mpi_request_ptr_; }

      constexpr reference operator*() const noexcept { return reference{*mpi_request_ptr_}; }
      constexpr pointer operator->() const noexcept { return mpi_request_ptr_; }

      constexpr request_array_iterator& operator++() noexcept { ++mpi_request_ptr_; return *this; }
      constexpr request_array_iterator operator++(int) noexcept { auto result = *this; ++mpi_request_ptr_; return result; }
      constexpr request_array_iterator& operator--() noexcept { --mpi_request_ptr_; return *this; }
      constexpr request_array_iterator operator--(int) noexcept { auto result = *this; --mpi_request_ptr_; return result; }
      constexpr request_array_iterator& operator+=(difference_type const n) noexcept { mpi_request_ptr_ += n; return *this; }
      constexpr request_array_iterator& operator-=(difference_type const n) noexcept { mpi_request_ptr_ -= n; return *this; }
      constexpr difference_type operator-(request_array_iterator const& other) noexcept { return mpi_request_ptr_ - other.mpi_request_ptr_; }
      constexpr reference operator[](difference_type const n) const { return reference{mpi_request_ptr_[n]}; }
    };

    template <typename Request>
    class request_array_const_iterator
    {
      MPI_Request const* mpi_request_ptr_;

     public:
      using value_type = Request;
      using pointer = MPI_Request const*;
      using reference = typename Request::const_reference;
      using difference_type = std::ptrdiff_t;
      using iterator_category = std::random_access_iterator_tag;

      constexpr request_array_const_iterator() noexcept = default;
      constexpr request_array_const_iterator(MPI_Request const& mpi_request) noexcept : mpi_request_ptr_{std::addressof(mpi_request)} { }
      constexpr request_array_const_iterator(request_array_iterator const& other) noexcept : mpi_request_ptr_{other.mpi_request_ptr_} { }

      constexpr bool operator==(request_array_const_iterator const& other) const noexcept { return mpi_request_ptr_ == other.mpi_request_ptr_; }
      constexpr bool operator<(request_array_const_iterator const& other) const noexcept { return mpi_request_ptr_ < other.mpi_request_ptr_; }

      constexpr reference operator*() const noexcept { return reference{*mpi_request_ptr_}; }
      constexpr pointer operator->() const noexcept { return mpi_request_ptr_; }

      constexpr request_array_const_iterator& operator++() noexcept { ++mpi_request_ptr_; return *this; }
      constexpr request_array_const_iterator operator++(int) noexcept { auto result = *this; ++mpi_request_ptr_; return result; }
      constexpr request_array_const_iterator& operator--() noexcept { --mpi_request_ptr_; return *this; }
      constexpr request_array_const_iterator operator--(int) noexcept { auto result = *this; --mpi_request_ptr_; return result; }
      constexpr request_array_const_iterator& operator+=(difference_type const n) noexcept { mpi_request_ptr_ += n; return *this; }
      constexpr request_array_const_iterator& operator-=(difference_type const n) noexcept { mpi_request_ptr_ -= n; return *this; }
      constexpr difference_type operator-(request_array_const_iterator const& other) noexcept { return mpi_request_ptr_ - other.mpi_request_ptr_; }
      constexpr difference_type operator-(request_array_iterator const& other) noexcept { return mpi_request_ptr_ - other.mpi_request_ptr_; }
      constexpr reference operator[](difference_type const n) const { return reference{mpi_request_ptr_[n]}; }
    };

    template <typename Request, std::size_t N>
    class request_array_base
    {
     protected:
      MPI_Request data_[N];

     public:
      using value_type = Request;
      using size_type = std::size_t;
      using difference_type = std::ptrdiff_t;
      using reference = typename Request::reference;
      using const_reference = typename Request::const_reference;
      using pointer = MPI_Request*;
      using const_pointer = MPI_Request const*;
      using iterator = ::yampi::request_array_detail::request_array_iterator<Request>;
      using const_iterator = ::yampi::request_array_detail::request_array_const_iterator<Request>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;

      bool operator==(request_array_base const& other) { return std::equal(data_, data_ + N, other.data_); }

      constexpr reference at(size_type const position)
      {
        if (position >= N)
          throw std::out_of_range("out of range");
        return reference(data_[position]);
      }

      constexpr const_reference at(size_type const position) const
      {
        if (position >= N)
          throw std::out_of_range("out of range");
        return const_reference(data_[position]);
      }

      constexpr reference operator[](size_type const position)
      { assert(position < N); return reference(data_[position]); }

      constexpr const_reference operator[](size_type const position) const
      { assert(position < N); return const_reference(data_[position]); }

      constexpr reference front() { return reference(data_[0u]); }
      constexpr const_reference front() const { return const_reference(data_[0u]); }
      constexpr reference back() { return reference(data_[N - 1u]); }
      constexpr const_reference back() const { return const_reference(data_[N - 1u]); }
      constexpr pointer data() noexcept { return data_; }
      constexpr const_pointer data() const noexcept { return data_; }

      constexpr iterator begin() noexcept { return {data_}; }
      constexpr const_iterator begin() const noexcept { return {data_}; }
      constexpr const_iterator cbegin() const noexcept { return {data_}; }
      constexpr iterator end() noexcept { return {data_ + N}; }
      constexpr const_iterator end() const noexcept { return {data_ + N}; }
      constexpr const_iterator cend() const noexcept { return {data_ + N}; }
      constexpr reverse_iterator rbegin() noexcept { return {this->end()}; }
      constexpr const_reverse_iterator rbegin() const noexcept { return {this->end()}; }
      constexpr const_reverse_iterator crbegin() const noexcept { return {this->cend()}; }
      constexpr reverse_iterator rend() noexcept { return {this->begin()}; }
      constexpr const_reverse_iterator rend() const noexcept { return {this->begin()}; }
      constexpr const_reverse_iterator crend() const noexcept { return {this->cbegin()}; }

      constexpr bool empty() const noexcept { return false; }
      constexpr size_type size() const noexcept { return N; }
      constexpr size_type max_size() const noexcept { return N; }

      std::pair< ::yampi::status, size_type > wait_any(::yampi::environment const& environment)
      {
        MPI_Status mpi_status;
        int index;
        int const error_code = MPI_Waitany(N, data_, std::addressof(index), std::addressof(mpi_status));

        return error_code == MPI_SUCCESS
          ? index != MPI_UNDEFINED
            ? std::make_pair(::yampi::status(mpi_status), static_cast<size_type>(index))
            : std::make_pair(::yampi::status(mpi_status), N)
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_any", environment);
      }

      size_type wait_any(::yampi::ignore_status_t const, ::yampi::environment const& environment)
      {
        int index;
        int const error_code = MPI_Waitany(N, data_, std::addressof(index), MPI_STATUS_IGNORE);

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
        int const error_code = MPI_Testany(N, data_, std::addressof(index), std::addressof(flag), std::addressof(mpi_status));

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
        int const error_code = MPI_Testany(N, data_, std::addressof(index), std::addressof(flag), MPI_STATUS_IGNORE);

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
          (std::is_same<
             typename std::remove_cv<
               typename std::iterator_traits<ContiguousIterator>::value_type>::type,
             ::yampi::status>::value),
          "Value type of ContiguousIterator must be the same to ::yampi::status");

        MPI_Status mpi_statuses[N];
        int const error_code = MPI_Waitall(N, data_, mpi_statuses);

        std::transform(
          mpi_statuses, mpi_statuses + N, out,
          [](MPI_Status const& mpi_status) { return ::yampi::status(mpi_status); });

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
          (std::is_same<
             typename std::remove_cv<
               typename std::iterator_traits<ContiguousIterator>::value_type>::type,
             ::yampi::status>::value),
          "Value type of ContiguousIterator must be the same to ::yampi::status");

        MPI_Status mpi_statuses[N];
        int flag;
        int const error_code = MPI_Testall(N, data_, std::addressof(flag), mpi_statuses);

        std::transform(
          mpi_statuses, mpi_statuses + N, out,
          [](MPI_Status const& mpi_status) { return ::yampi::status(mpi_status); });

        return error_code == MPI_SUCCESS
          ? std::make_pair(static_cast<bool>(flag), out)
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_all", environment);
      }

      bool test_all(::yampi::environment const& environment)
      {
        int flag;
        int const error_code = MPI_Testall(N, data_, std::addressof(flag), MPI_STATUSES_IGNORE);
        return error_code == MPI_SUCCESS
          ? static_cast<bool>(flag)
          : throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_all", environment);
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2>
      std::pair<bool, ContiguousIterator1> wait_some(
        ContiguousIterator1 const out_index, ContiguousIterator2 const out_status, ::yampi::environment const& environment)
      {
        static_assert(
          (std::is_same<
             typename std::remove_cv<
               typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
             size_type>::value),
          "Value type of ContiguousIterator1 must be the same to size_type");
        static_assert(
          (std::is_same<
             typename std::remove_cv<
               typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
             ::yampi::status>::value),
          "Value type of ContiguousIterator2 must be the same to ::yampi::status");

        int indices[N];
        MPI_Status mpi_statuses[N];
        int num_completed_requests;
        int const error_code = MPI_Waitsome(N, data_, std::addressof(num_completed_requests), indices, mpi_statuses);

        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_some", environment);

        if (num_completed_requests == MPI_UNDEFINED)
          return std::make_pair(false, out_index);

        std::transform(
          indices, indices + num_completed_requests, out_index,
          [](int const index) { return static_cast<size_type>(index); });
        std::transform(
          mpi_statuses, mpi_statuses + num_completed_requests, out_status,
          [](MPI_Status const& mpi_status) { return ::yampi::status(mpi_status); });

        return std::make_pair(true, out_index);
      }

      template <typename ContiguousIterator>
      std::pair<bool, ContiguousIterator> wait_some(
        ContiguousIterator const out_index, ::yampi::environment const& environment)
      {
        static_assert(
          (std::is_same<
             typename std::remove_cv<
               typename std::iterator_traits<ContiguousIterator>::value_type>::type,
             size_type>::value),
          "Value type of ContiguousIterator must be the same to size_type");

        int indices[N];
        int num_completed_requests;
        int const error_code = MPI_Waitsome(N, data_, std::addressof(num_completed_requests), indices, MPI_STATUSES_IGNORE);

        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::wait_some", environment);

        if (num_completed_requests == MPI_UNDEFINED)
          return std::make_pair(false, out_index);

        std::transform(
          indices, indices + num_completed_requests, out_index,
          [](int const index) { return static_cast<size_type>(index); });

        return std::make_pair(true, out_index);
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2>
      std::pair<bool, ContiguousIterator1> test_some(
        ContiguousIterator1 const out_index, ContiguousIterator2 const out_status, ::yampi::environment const& environment)
      {
        static_assert(
          (std::is_same<
             typename std::remove_cv<
               typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
             size_type>::value),
          "Value type of ContiguousIterator1 must be the same to size_type");
        static_assert(
          (std::is_same<
             typename std::remove_cv<
               typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
             ::yampi::status>::value),
          "Value type of ContiguousIterator2 must be the same to ::yampi::status");

        int indices[N];
        MPI_Status mpi_statuses[N];
        int num_completed_requests;
        int const error_code = MPI_Testsome(N, data_, std::addressof(num_completed_requests), indices, mpi_statuses);

        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_some", environment);

        if (num_completed_requests == MPI_UNDEFINED)
          return std::make_pair(false, out_index);

        std::transform(
          indices, indices + num_completed_requests, out_index,
          [](int const index) { return static_cast<size_type>(index); });
        std::transform(
          mpi_statuses, mpi_statuses + num_completed_requests, out_status,
          [](MPI_Status const& mpi_status) { return ::yampi::status(mpi_status); });

        return std::make_pair(true, out_index);
      }

      template <typename ContiguousIterator>
      std::pair<bool, ContiguousIterator> test_some(
        ContiguousIterator const out_index, ::yampi::environment const& environment)
      {
        static_assert(
          (std::is_same<
             typename std::remove_cv<
               typename std::iterator_traits<ContiguousIterator>::value_type>::type,
             size_type>::value),
          "Value type of ContiguousIterator must be the same to size_type");

        int indices[N];
        int num_completed_requests;
        int const error_code = MPI_Testsome(N, data_, std::addressof(num_completed_requests), indices, MPI_STATUSES_IGNORE);

        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::request_array_detail::request_array_base::test_some", environment);

        if (num_completed_requests == MPI_UNDEFINED)
          return std::make_pair(false, out_index);

        std::transform(
          indices, indices + num_completed_requests, out_index,
          [](int const index) { return static_cast<size_type>(index); });

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


#endif

