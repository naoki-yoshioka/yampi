#ifndef YAMPI_INFORMATION_HPP
# define YAMPI_INFORMATION_HPP

# include <boost/config.hpp>

# include <string>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_assign.hpp>
#   include <boost/type_traits/is_nothrow_move_constructible.hpp>
#   include <boost/type_traits/is_nothrow_move_assignable.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_copy_assignable boost::has_nothrow_assign
#   define YAMPI_is_nothrow_move_constructible boost::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable boost::is_nothrow_move_assignable
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
  class information
  {
    MPI_Info mpi_info_;

   public:
    information()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Info>::value)
      : mpi_info_(MPI_INFO_NULL)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    information(information const&) = delete;
    information& operator=(information const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    information(information const&);
    information& operator=(information const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    information(information&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Info>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Info>::value)
      : mpi_info_(std::move(other.mpi_info_))
    { other.mpi_info_ = MPI_INFO_NULL; }

    information& operator=(information&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Info>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Info>::value)
    {
      if (this != YAMPI_addressof(other))
      {
        if (mpi_info_ != MPI_INFO_NULL)
          MPI_Info_free(YAMPI_addressof(mpi_info_));
        mpi_info_ = std::move(other.mpi_info_);
        other.mpi_info_ = MPI_INFO_NULL;
      } 
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~information() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_info_ == MPI_INFO_NULL)
        return;

      MPI_Info_free(YAMPI_addressof(mpi_info_));
    }

    explicit information(MPI_Info const& mpi_info)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Group>::value)
      : mpi_info_(mpi_info)
    { }

    explicit information(::yampi::environment const& environment)
      : mpi_info_(create(environment))
    { }

    information(information const& other, ::yampi::environment const& environment)
      : mpi_info_(duplicate(other, environment))
    { }

   private:
    MPI_Info create(::yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Info_create(YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::information::create", environment);
    }

    MPI_Info duplicate(
      information const& other, ::yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Info_dup(other.mpi_info_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::information::duplicate", environment);
    }

   public:
    void reset(MPI_Info const& mpi_info, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_info_ = mpi_info;
    }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void reset(information&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_info_ = std::move(other.mpi_info_);
      other.mpi_info_ = MPI_INFO_NULL;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    void reset(::yampi::environment const& environment)
    {
      free(environment);
      mpi_info_ = create(environment);
    }

    void reset(information const& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_info_ = duplicate(other, environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_info_ == MPI_INFO_NULL)
        return;

      int const error_code = MPI_Info_free(YAMPI_addressof(mpi_info_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::free", environment);
    }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_info_ == MPI_INFO_NULL))
    { return mpi_info_ == MPI_INFO_NULL; }

    void insert(
      std::string const& key, std::string const& value,
      ::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 3
      int const error_code = MPI_Info_set(mpi_info_, key.c_str(), value.c_str());
# else
      int const error_code
        = MPI_Info_set(
            mpi_info_, const_cast<char*>(key.c_str()), const_cast<char*>(value.c_str()));
# endif
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::insert", environment);
    }

    void erase(std::string const& key, ::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 3
      int const error_code = MPI_Info_delete(mpi_info_, key.c_str());
# else
      int const error_code = MPI_Info_delete(mpi_info_, const_cast<char*>(key.c_str()));
# endif
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::erase", environment);
    }

    boost::optional<std::string> at(
      std::string const& key, ::yampi::environment const& environment) const
    {
      int value_length;
      int flag;
# if MPI_VERSION >= 3
      int error_code
        = MPI_Info_get_valuelen(
            mpi_info_, key.c_str(),
            YAMPI_addressof(value_length), YAMPI_addressof(flag));
# else
      int error_code
        = MPI_Info_get_valuelen(
            mpi_info_, const_cast<char*>(key.c_str()),
            YAMPI_addressof(value_length), YAMPI_addressof(flag));
# endif
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::get", environment);

      if (flag == false)
        return boost::none;

      std::string result(value_length, ' ');
# if MPI_VERSION >= 3
      error_code
        = MPI_Info_get(
            mpi_info_, key.c_str(),
            value_length, const_cast<char*>(result.c_str()), YAMPI_addressof(flag));
# else
      error_code
        = MPI_Info_get(
            mpi_info_, const_cast<char*>(key.c_str()),
            value_length, const_cast<char*>(result.c_str()), YAMPI_addressof(flag));
# endif
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::get", environment);

      if (flag == false)
        return boost::none;

      return boost::make_optional(result);
    }

    int num_keys(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Info_get_nkeys(mpi_info_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::information::num_keys", environment);
    }

    std::string key(int const n, ::yampi::environment const& environment) const
    {
      char key[MPI_MAX_INFO_KEY];
      int const error_code = MPI_Info_get_nthkey(mpi_info_, n, key);
      return error_code == MPI_SUCCESS
        ? std::string(key)
        : throw ::yampi::error(error_code, "yampi::information::key", environment);
    }

    MPI_Info const& mpi_info() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_info_; }

    void swap(information& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Info>::value)
    {
      using std::swap;
      swap(mpi_info_, other.mpi_info_);
    }
  };

  inline void swap(::yampi::information& lhs, ::yampi::information& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif

