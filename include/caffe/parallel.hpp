#ifndef CAFFE_PARALLEL_H_
#define CAFFE_PARALLEL_H_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  explicit Params(const Solver<Dtype>& root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }

 protected:
  const size_t size_;           // Size of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient

DISABLE_COPY_AND_ASSIGN(Params);
};

#ifndef CPU_ONLY

// Params for GPU memory.
template<typename Dtype>
class GPUParams : public Params<Dtype> {
 public:
  GPUParams(const Solver<Dtype>& root_solver, int device);
  virtual ~GPUParams();

  void configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

// Pair of GPUs, used to build a tree that maps the machine's topology
class DevicePair {
 public:
  DevicePair(int parent, int device)
      : parent(parent),
        device(device) {
  }

  int parent;
  int device;

  // Group GPUs in pairs, by proximity for better map-reduce efficiency
  static void compute(const vector<int> devices, vector<DevicePair>* pairs);
};

// Synchronous data parallelism using P2P GPU transfers.
template<typename Dtype>
class P2PSync :  //
    public GPUParams<Dtype>,  //
    public Solver<Dtype>::Callback,  //
    public InternalThread {
 public:
  explicit P2PSync(Solver<Dtype>* root_solver, P2PSync<Dtype>* parent,
                   const SolverParameter& param);
  virtual ~P2PSync();

  inline const P2PSync<Dtype>* parent() const {
    return parent_;
  }
  inline void add_child(shared_ptr<P2PSync<Dtype> > value) {
    children_.push_back(value);
  }
  inline const Solver<Dtype>* solver() const {
    return solver_;
  }

 protected:
  void before_iteration(Timer* timer, ostringstream* timing);
  void finish_iteration(Timer* timer, ostringstream* timing);

  void InternalThreadEntry();

  P2PSync<Dtype>* parent_;
  vector<shared_ptr<P2PSync<Dtype> > > children_;
  blocking_queue<P2PSync<Dtype>*> queue_;
  const int solver_count_;
  const int initial_iter_;
  Dtype* parent_grads_;
  Solver<Dtype>* solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

#endif

}  // namespace caffe

#endif
