#ifndef YAMPI_TARGET_BUFFER_HPP
# define YAMPI_TARGET_BUFFER_HPP

# include <boost/config.hpp>

# include <cassert>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/is_integral.hpp>
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <yampi/datatype.hpp>
# include <yampi/predefined_datatype.hpp>
# include <yampi/has_predefined_datatype.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_integral std::is_integral
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_is_integral boost::is_integral
#   define YAMPI_enable_if boost::enable_if_c
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif


namespace yampi
{
  template <typename T, typename Enable = void>
  class target_buffer
  {
    MPI_Aint mpi_displacement_;
    int count_;
    ::yampi::datatype const* datatype_ptr_;

   public:
    template <typename Integer>
    target_buffer(Integer const displacement, ::yampi::datatype const& datatype) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_displacement_(static_cast<MPI_Aint>(displacement)), count_(1),
        datatype_ptr_(YAMPI_addressof(datatype))
    {
      static_assert(YAMPI_is_integral<Integer>::value, "Integer should be an integral type");
      assert(displacement >= Integer{0});
    }

    template <typename Integer>
    target_buffer(Integer const displacement, int const count, ::yampi::datatype const& datatype) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_displacement_(static_cast<MPI_Aint>(displacement)), count_(count),
        datatype_ptr_(YAMPI_addressof(datatype))
    {
      static_assert(YAMPI_is_integral<Integer>::value, "Integer should be an integral type");
      assert(displacement >= Integer{0});
      assert(count >= 0);
    }

    bool operator==(target_buffer const& other) const BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(datatype_ == other.datatype_))
    { return mpi_displacement_ == other.mpi_displacement_ and count_ == other.count_ and *datatype_ptr_ == *other.datatype_ptr_; }

    MPI_Aint const& mpi_displacement() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_displacement_; }
    int const& count() const BOOST_NOEXCEPT_OR_NOTHROW { return count_; }
    ::yampi::datatype const& datatype() const BOOST_NOEXCEPT_OR_NOTHROW { return *datatype_ptr_; }

    void swap(target_buffer& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_displacement_, other.mpi_displacement_);
      swap(count_, other.count_);
      swap(datatype_ptr_, other.datatype_ptr_);
    }
  }; // class target_buffer<T, Enable>

  template <typename T>
  class target_buffer<T, typename YAMPI_enable_if< ::yampi::has_predefined_datatype<T>::value >::type>
  {
    MPI_Aint mpi_displacement_;
    int count_;

   public:
    template <typename Integer>
    explicit target_buffer(Integer const displacement) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_displacement_(static_cast<MPI_Aint>(displacement)), count_(1)
    {
      static_assert(YAMPI_is_integral<Integer>::value, "Integer should be an integral type");
      assert(displacement >= Integer{0});
    }

    template <typename Integer>
    target_buffer(Integer const displacement, int const count) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_displacement_(static_cast<MPI_Aint>(displacement)), count_(count)
    {
      static_assert(YAMPI_is_integral<Integer>::value, "Integer should be an integral type");
      assert(displacement >= Integer{0});
      assert(count >= 0);
    }

    bool operator==(target_buffer const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_displacement_ == other.mpi_displacement_ and count_ == other.count_; }

    MPI_Aint const& mpi_displacement() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_displacement_; }
    int const& count() const BOOST_NOEXCEPT_OR_NOTHROW { return count_; }
    ::yampi::predefined_datatype<T> datatype() const BOOST_NOEXCEPT_OR_NOTHROW { return ::yampi::predefined_datatype<T>(); }

    void swap(target_buffer& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_displacement_, other.mpi_displacement_);
      swap(count_, other.count_);
    }
  }; // class target_buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>

  template <typename T>
  inline bool operator!=(::yampi::target_buffer<T> const& lhs, ::yampi::target_buffer<T> const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  template <typename T>
  inline void swap(::yampi::target_buffer<T>& lhs, ::yampi::target_buffer<T>& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  template <typename T, typename Integer>
  inline
  typename YAMPI_enable_if< ::yampi::has_predefined_datatype<T>::value, ::yampi::target_buffer<T> >::type make_target_buffer(Integer const displacement)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(::yampi::target_buffer<T>(displacement)))
  { return ::yampi::target_buffer<T>(displacement); }

  template <typename T, typename Integer>
  inline ::yampi::target_buffer<T> make_target_buffer(Integer const displacement, ::yampi::predefined_datatype<T> const&)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(::yampi::target_buffer<T>(displacement)))
  { return ::yampi::target_buffer<T>(displacement); }

  template <typename T, typename Integer>
  inline ::yampi::target_buffer<T> make_target_buffer(Integer const displacement, ::yampi::datatype const& datatype)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(::yampi::target_buffer<T>(displacement, datatype)))
  { return ::yampi::target_buffer<T>(displacement, datatype); }

  template <typename T, typename Integer>
  inline
  typename YAMPI_enable_if< ::yampi::has_predefined_datatype<T>::value, ::yampi::target_buffer<T> >::type make_target_buffer(Integer const displacement, int count)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(::yampi::target_buffer<T>(displacement, count)))
  { return ::yampi::target_buffer<T>(displacement, count); }

  template <typename T, typename Integer>
  inline ::yampi::target_buffer<T> make_target_buffer(Integer const displacement, int count, ::yampi::predefined_datatype<T> const&)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(::yampi::target_buffer<T>(displacement, count)))
  { return ::yampi::target_buffer<T>(displacement, count); }

  template <typename T, typename Integer>
  inline ::yampi::target_buffer<T> make_target_buffer(Integer const displacement, int count, ::yampi::datatype const& datatype)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(::yampi::target_buffer<T>(displacement, count, datatype)))
  { return ::yampi::target_buffer<T>(displacement, count, datatype); }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_enable_if
# undef YAMPI_is_integral

#endif
