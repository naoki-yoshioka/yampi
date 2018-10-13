#ifndef YAMPI_INFORMATION_HPP
# define YAMPI_INFORMATION_HPP

# include <boost/config.hpp>

# include <string>
# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>

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
    information() : mpi_info_(MPI_INFO_NULL) { }
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
      : mpi_info_(std::move(other.mpi_info_))
    { other.mpi_info_ = MPI_INFO_NULL; }

    information& operator=(information&& other)
    {
      if (this != YAMPI_addressof(other))
      {
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

    MPI_Info const& mpi_info() const { return mpi_info_; }
    void mpi_info(MPI_Info const& info) { mpi_info_ = info; }
  };
}


# undef YAMPI_addressof

#endif

