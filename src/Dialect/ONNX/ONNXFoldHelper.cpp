#include "src/Dialect/ONNX/ONNXFoldHelper.hpp"

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for binary in presence of broadcast.
//===----------------------------------------------------------------------===//

// Template to generate binary operation results. It takes as inupt
// the element type as well as the two element attributes for the
// operation, and return the result of the operation, also as an
// attribute.

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXAddOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    APFloat lhsVal = lhsAttr.cast<FloatAttr>().getValue();
    APFloat rhsVal = secondAttr.cast<FloatAttr>().getValue();
    APFloat res = lhsVal + rhsVal;
    // Could use the APFloat interface to emulate the results, are ok to simply
    // perform them in the highest possible precision.
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt lhsVal = lhsAttr.cast<IntegerAttr>().getValue();
    APInt rhsVal = secondAttr.cast<IntegerAttr>().getValue();
    APInt res = lhsVal + rhsVal;
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for AddOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXSubOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    APFloat lhsVal = lhsAttr.cast<FloatAttr>().getValue();
    APFloat rhsVal = secondAttr.cast<FloatAttr>().getValue();
    APFloat res = lhsVal - rhsVal;
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt lhsVal = lhsAttr.cast<IntegerAttr>().getValue();
    APInt rhsVal = secondAttr.cast<IntegerAttr>().getValue();
    APInt res = lhsVal - rhsVal;
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for SubOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXMulOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    APFloat lhsVal = lhsAttr.cast<FloatAttr>().getValue();
    APFloat rhsVal = secondAttr.cast<FloatAttr>().getValue();
    APFloat res = lhsVal * rhsVal;
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt lhsVal = lhsAttr.cast<IntegerAttr>().getValue();
    APInt rhsVal = secondAttr.cast<IntegerAttr>().getValue();
    APInt res = lhsVal * rhsVal;
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for MulOp: unkonwn data type");
}

namespace {
APInt divideAPInts(const IntegerType type, const APInt a, const APInt b) {
  if (type.isUnsigned()) {
    return a.udiv(b);
  } else { // Signed or Signless are both treated as Signed in Onnx-Mlir
    return a.sdiv(b);
  }
}
} // namespace

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXDivOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    APFloat lhsVal = lhsAttr.cast<FloatAttr>().getValue();
    APFloat rhsVal = secondAttr.cast<FloatAttr>().getValue();
    APFloat res = lhsVal / rhsVal;
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt lhsVal = lhsAttr.cast<IntegerAttr>().getValue();
    APInt rhsVal = secondAttr.cast<IntegerAttr>().getValue();
    APInt res = divideAPInts(elementType.cast<IntegerType>(), lhsVal, rhsVal);
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for DivOp: unkonwn data type");
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unary operation.
//===----------------------------------------------------------------------===//

template <>
Attribute ComputeConstPropElementwiseUnary<ONNXNegOp>(
    Builder &builder, Type elementType, Attribute attr) {
  if (elementType.isa<FloatType>()) {
    APFloat val = attr.cast<FloatAttr>().getValue();
    APFloat res = -val;
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt val = attr.cast<IntegerAttr>().getValue();
    APInt res = -val;
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for NegOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseUnary<ONNXSqrtOp>(
    Builder &builder, Type elementType, Attribute attr) {
  if (elementType.isa<FloatType>()) {
    double val = attr.cast<FloatAttr>().getValueAsDouble();
    double res = sqrt(val);
    return builder.getFloatAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for SqrtOp: unkonwn data type");
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for transpose.
//===----------------------------------------------------------------------===//

static void RecurseConstPropTranspose(Builder &builder,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, SmallVector<uint64_t, 4> &perm,
    int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    auto res = attr.getValue(ArrayRef<uint64_t>(indices));
    resVector.emplace_back(res);
  } else {
    // Recurse.
    auto shape = attr.getType().getShape();
    int rank = shape.size();
    int index = perm[rank - freeRank];
    int size = attr.getType().getShape()[index];
    for (int i = 0; i < size; ++i) {
      indices[index] = i;
      RecurseConstPropTranspose(
          builder, resVector, attr, indices, perm, freeRank - 1);
    }
  }
}

DenseElementsAttr ConstPropTranspose(
    Builder &builder, Value resOperand, Attribute attr, ArrayAttr permAttr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  auto rank = denseAttr.getType().getShape().size();
  // Read permute vector.
  SmallVector<uint64_t, 4> perm;
  assert(permAttr && "permute attribute expected to be defined here");
  for (auto permVal : permAttr.getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());
  // Init indice vector.
  SmallVector<uint64_t, 4> indices(rank, 0);
  std::vector<Attribute> resVector;
  // Copy using permute order.
  RecurseConstPropTranspose(builder, resVector, denseAttr, indices, perm, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unsqueeze.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropUnsqueeze(
    Builder &builder, Value resOperand, Attribute attr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();

  // Unqueeze does not change the order of access, so just copy the whole data.
  std::vector<Attribute> resVector;
  for (auto value : denseAttr.getValues<Attribute>()) {
    resVector.emplace_back(value);
  }

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for squeeze.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropSqueeze(
    Builder &builder, Value resOperand, Attribute attr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();

  // Squeeze does not change the order of access, so just copy the whole data.
  std::vector<Attribute> resVector;
  for (auto value : denseAttr.getValues<Attribute>()) {
    resVector.emplace_back(value);
  }

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for gather.
//===----------------------------------------------------------------------===//

// Assume input(data) is a rank r and indices is a rank q tesor.
// input[i_0 x i_1 x ... x i_r-1] indices[j_0 x j_1 x j_q-1]
// The result rank is q + (r-1). result shape is determined by replacing element
// "axis" of the input shape with indices shape. For example for axis = 3 the
// result shape would be:
// res [i_0 x i_1 x i_2 x j_0 x j_1 x ... j_q-1 x i_4 x i_5 x ... x i_r-1]
//
//               axis ────┐                   ┌─── axis + q
//                        ▼                   ▼
//                        ┌───────────────────┐
// Res [ k_0 x k_1 x ... x│k_i0 x ... x k_iq-1│x ... x k_r]
//                        └────────┬──────────┘
//                                 │
//                                 │ Replace with:
//                                 │
//                                 │ K = indices[k_i0 x ... x k_iq-1]
//                                 ▼
//                                ┌─┐
//      = input [k_0 x k_1 x ... x│K│x ...x k_r]
//                                └─┘

DenseElementsAttr ConstPropGather(
    Builder &builder, Value resOperand, Attribute data,
    Attribute indices, int64_t axis) {
  typedef SmallVector<uint64_t, 4> TensorShape;
  DenseElementsAttr inputAttr = data.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(inputAttr && "expected dense attribute");
  DenseElementsAttr indicesAttr =
      indices.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(indicesAttr && "expected dense attribute");

  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  std::vector<Attribute> resVector;

  auto inputShape = data.getType().cast<RankedTensorType>().getShape();
  auto indicesShape = indices.getType().cast<RankedTensorType>().getShape();
  auto resShape = resType.getShape();
  int nResElements = resType.getNumElements();

  // Given element at location "index" in a tensor with shape "shape", proceed
  // to the next element (in flat represention) by updating the "index"
  auto proceed = [] (TensorShape &index, const ArrayRef<long> &shape) {
    assert(index.size() == shape.size());
    if (shape.size() == 0)
      return;
    int dim = shape.size() - 1;
    index[dim]++;
    while(dim > 0) {
      if (index[dim] >= shape[dim]) {
        index[dim] = 0;
        index[dim - 1]++;
      }
      else
        break;
      dim--;
    }
  };

  // Return the index of an element in the flat represention of a tensor
  auto flatten = [] (const TensorShape &index, const ArrayRef<long> &shape) {
      assert(index.size() == shape.size());
      int dim = shape.size() - 1;
      int res = 0;
      int mul = 1;
      while(dim >= 0) {
        res += (mul * index[dim]);
        mul *= shape[dim];
        dim--;
      }
      return res;
  };

  std::vector<Attribute> indicesVector;
  for (auto elem : indicesAttr.getValues<Attribute>())
    indicesVector.push_back(elem);
  std::vector<Attribute> inputVector;
  for (auto elem : inputAttr.getValues<Attribute>())
    inputVector.push_back(elem);

  TensorShape resIndex(resShape.size(), 0);
  // Traverse all result elements
  for (int resFlatIndex = 0; resFlatIndex < nResElements; resFlatIndex++) {
    // Extract the indices location base on axis
    TensorShape indicesIndex(resIndex.begin() + axis,
               resIndex.begin() + axis + indicesShape.size());
    int flattenIndicesIndex = flatten(indicesIndex, indicesShape);
    // K is result of look up from indices tensor
    int k = indicesVector[flattenIndicesIndex].cast<IntegerAttr>().getInt();

    // Create a new tensor shape by replacing indices location with K
    TensorShape inputIndex(resIndex.begin(),
               resIndex.begin() + axis);
    inputIndex.push_back(k);
    inputIndex.insert(inputIndex.end(), resIndex.begin() + axis + indicesShape.size(), resIndex.end());

    // Look up the input and add to the result tensor
    int flattenInputIndex = flatten(inputIndex, inputShape);
    resVector.emplace_back(inputVector[flattenInputIndex]);

    // Proceed to the next output
    proceed(resIndex, resShape);
  }
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for constantOfShape.
//===----------------------------------------------------------------------===//

DenseElementsAttr ConstPropConstantOfShape(Builder &builder, Value resOperand,
    Attribute input, Attribute value) {
  DenseElementsAttr inputAttr = input.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(inputAttr && "expected dense attribute");
  DenseElementsAttr valueAttr = value.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(valueAttr && "expected dense attribute");
  auto resType = resOperand.getType().cast<RankedTensorType>();
  auto inputShape = input.getType().cast<RankedTensorType>().getShape();
  auto resShape = resType.getShape();
  int nResElements = resType.getNumElements();
  auto element = valueAttr.getValue<Attribute>(0);
  std::vector<Attribute> resVector(nResElements, element);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for relu.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropRelu(
    Builder &builder, Value resOperand, Attribute input) {
  DenseElementsAttr inputAttr = input.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(inputAttr && "expected dense attribute");
  auto resType = resOperand.getType().cast<RankedTensorType>();
  auto elementType = inputAttr.getType().getElementType();
  int nResElements = resType.getNumElements();
  std::vector<Attribute> resVector;
  
  for (auto element : inputAttr.getValues<Attribute>()) {
    if (elementType.isa<FloatType>()) {
      double value = element.cast<FloatAttr>().getValueAsDouble();
      Attribute res = builder.getFloatAttr(elementType, value > 0.0 ? value : 0.0);
      resVector.push_back(res);
    }
    // if (resType.isa<IntegerType>()) {
    //   APInt val = element.cast<IntegerAttr>().getValue();
    //   APInt res = val > 0 ? val : 0;
    //   resVector.push_back(builder.getIntegerAttr(resType, res));
    // }
  }
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for concat.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropConcat(
    Builder &builder, Value resOperand, ArrayRef<Attribute> attrs) {
  // TODO: expand this to support more than 1D Tensor
  std::vector<Attribute> resVector;
  for (auto attr : attrs) {
    // Read dense attribute, the constant tensor we are transforming.
    DenseElementsAttr denseAttr =
        attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
    assert(denseAttr && "expected dense attribute");

    for (auto value : denseAttr.getValues<Attribute>()) {
      resVector.emplace_back(value);
    }
  }
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  assert(
      resType.getShape()[0] == resVector.size() && "Unmatching operand size");

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for slice.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropSlice(Builder &builder, Value resOperand,
    Attribute data, Attribute starts, Attribute ends, Attribute axes,
    Attribute steps) {
  // TODO: expand this to support more than 1D Tensor
  //       Also fix restrictions on parameters

  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr dataAttr = data.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(dataAttr && "expected dense attribute");
  DenseElementsAttr startsAttr =
      starts.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(startsAttr && "expected dense attribute");
  DenseElementsAttr endsAttr = ends.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(endsAttr && "expected dense attribute");
  DenseElementsAttr axesAttr = axes.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(axesAttr && "expected dense attribute");
  DenseElementsAttr stepsAttr =
      steps.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(stepsAttr && "expected dense attribute");

  int64_t startLocal = startsAttr.getValue<IntegerAttr>(0).getInt();
  int64_t endLocal = endsAttr.getValue<IntegerAttr>(0).getInt();
  int64_t axisLocal = axesAttr.getValue<IntegerAttr>(0).getInt();
  int64_t stepLocal = stepsAttr.getValue<IntegerAttr>(0).getInt();
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  auto dataSize = dataAttr.size();

  if (startLocal < 0) {
    startLocal = startLocal + dataSize;
    if (startLocal < 0)
      startLocal = 0;
  } else if (startLocal > dataSize) {
    startLocal = dataSize;
  }
  if (endLocal < 0) {
    endLocal = endLocal + dataSize;
    if (endLocal < 0)
      endLocal = 0;
  } else if (endLocal > dataSize) {
    endLocal = dataSize;
  }
  int64_t nElements = resType.getNumElements();
  assert(((endLocal - startLocal) * stepLocal >= 0) && axisLocal == 0  &&
    nElements >= 0 && "Unsupported slice operation mode");

  std::vector<Attribute> resVector;

  auto element = dataAttr.attr_value_begin() + startLocal;
  for (int i = 0; i < nElements; i++) {
    resVector.emplace_back(*element);
    element += stepLocal;
  }

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

namespace {
APInt castIntToInt(APInt inVal, IntegerType fromType, IntegerType toType) {
  unsigned toWidth = toType.getWidth();
  if (fromType.isUnsigned() || toType.isUnsigned()) {
    // If either fromType or toType is unsigned, then we're expecting a
    // positive value, hence zero-extention will always be correct.
    return inVal.zextOrTrunc(toWidth);
  } else {
    // If both fromType and toType are signed, then sign-extension should
    // yield correct result.
    return inVal.sextOrTrunc(toWidth);
  }
}
} // namespace

DenseElementsAttr ConstPropCastIntToInt(
    Builder &builder, Value constOp, Attribute input, IntegerAttr to) {

  mlir::RankedTensorType constType =
      constOp.getType().cast<mlir::RankedTensorType>();
  IntegerType fromElemType = constType.getElementType().cast<IntegerType>();

  auto toAttr = to.getValue().getSExtValue();
  IntegerType toElemType = mlir::UnrankedTensorType::get(
      convertONNXTypeToMLIRType(
          builder, static_cast<onnx::TensorProto_DataType>(toAttr)))
                               .getElementType()
                               .cast<IntegerType>();

  assert(fromElemType.isa<IntegerType>() && toElemType.isa<IntegerType>());

  auto inputElems = input.cast<mlir::DenseElementsAttr>();
  std::vector<Attribute> result;

  for (IntegerAttr inputElement : inputElems.getValues<IntegerAttr>()) {
    APInt inVal = inputElement.getValue();
    APInt outVal = castIntToInt(inVal, fromElemType, toElemType);
    IntegerAttr attr = builder.getIntegerAttr(toElemType, outVal);
    result.push_back(attr);
  }

  auto constShape = constType.getShape();
  auto resultType = mlir::RankedTensorType::get(constShape, toElemType);
  auto resultAttr =
      DenseElementsAttr::get(resultType, llvm::makeArrayRef(result));
  return resultAttr;
}

bool canConstPropCastIntToInt(
    Builder &builder, Value constOp, Attribute input, IntegerAttr to) {
  mlir::RankedTensorType constType =
      constOp.getType().cast<mlir::RankedTensorType>();
  Type fromElemType = constType.getElementType();

  auto toAttr = to.getValue().getSExtValue();
  auto toElemType = mlir::UnrankedTensorType::get(
      convertONNXTypeToMLIRType(
          builder, static_cast<onnx::TensorProto_DataType>(toAttr)))
                        .getElementType();

  return fromElemType.isa<IntegerType>() && toElemType.isa<IntegerType>();
}

bool isConstOfZeros(Builder &builder, Attribute attr) {

  DenseElementsAttr denseAttr = attr.cast<DenseElementsAttr>();
  mlir::Type constElemType = denseAttr.getType().getElementType();
  if (constElemType.isa<IntegerType>()) {
    for (IntegerAttr value : denseAttr.getValues<IntegerAttr>()) {
      APInt inVal = value.getValue();
      if (!inVal.isNullValue()) {
        return false;
      }
    }
  } else if (constElemType.isa<FloatType>()) {
    for (FloatAttr value : denseAttr.getValues<FloatAttr>()) {
      APFloat inVal = value.getValue();
      if (!inVal.isZero()) {
        return false;
      }
    }
  } else {
    return false;
  }

  return true;
}

DenseElementsAttr CreateZerosFromTemplate(
    Builder &builder, Value templateTensor) {
  ShapedType shapedType = templateTensor.getType().cast<ShapedType>();
  DenseElementsAttr resultAttr = DenseElementsAttr::get(shapedType, 0);
  return resultAttr;
}

namespace {
class BroadcastMatMulCalculator {
public:
  BroadcastMatMulCalculator(Value resultValue, ShapedType resultType,
      ArrayRef<int64_t> resultShape, DenseElementsAttr lhs,
      DenseElementsAttr rhs, std::vector<int64_t> normalizedLhsShape,
      std::vector<int64_t> normalizedRhsShape, int64_t m, int64_t k, int64_t n)
      : _resultValue(resultValue), _resultType(resultType),
        _resultShape(resultShape), _lhs(lhs), _rhs(rhs),
        _lhsShape(std::move(normalizedLhsShape)),
        _rhsShape(std::move(normalizedRhsShape)), _M(m), _K(k), _N(n),
        _dimOffsets(calculateDimOffsets(resultShape)) {}

  std::vector<APInt> calc() {
    _resultStorage = createInitialResultStorage(_resultType);
    recurse(0, 0);
    return std::move(_resultStorage);
  }

private:
  void recurse(size_t rank, size_t flattenedIdxBase) {
    int64_t lhsRank = _lhs.getType().getRank();
    int64_t rhsRank = _rhs.getType().getRank();
    IntegerType lhsEType = _lhs.getType().getElementType().cast<IntegerType>();
    IntegerType rhsEType = _rhs.getType().getElementType().cast<IntegerType>();
    if (rank + 2 == _lhsShape.size()) {
      // Do 2D MatMul stuff
      const IntegerType resultElementType =
          _resultType.getElementType().cast<IntegerType>();
      for (uint64_t i = 0; (int64_t)i < _M; ++i) {
        for (uint64_t j = 0; (int64_t)j < _N; ++j) {
          const uint64_t flattenedIdx = flattenedIdxBase + i * _N + j;
          for (uint64_t k = 0; (int64_t)k < _K; ++k) {
            auto lhsIdx = getBroadcastIdx(_idxStack, i, k, _lhsShape, lhsRank);
            APInt aRaw = _lhs.getValue(lhsIdx).cast<IntegerAttr>().getValue();
            auto rhsIdx = getBroadcastIdx(_idxStack, k, j, _rhsShape, rhsRank);
            APInt bRaw = _rhs.getValue(rhsIdx).cast<IntegerAttr>().getValue();
            APInt a = castIntToInt(aRaw, lhsEType, resultElementType);
            APInt b = castIntToInt(bRaw, rhsEType, resultElementType);
            APInt ab = a * b;
            _resultStorage.at(flattenedIdx) += ab;
          }
        }
      }
    } else {
      for (int64_t i = 0; i < _resultShape[rank]; ++i) {
        size_t nextRankIdx = flattenedIdxBase + i * _dimOffsets.at(rank + 1);
        _idxStack.push_back(i);
        recurse(rank + 1, nextRankIdx);
        _idxStack.pop_back();
      }
    }
  }

  static llvm::SmallVector<uint64_t, 6> getBroadcastIdx(
      const std::vector<int64_t> &idxStack, int64_t a, int64_t b,
      ArrayRef<int64_t> normShape, int64_t rank) {
    assert(rank >= 2);
    const int64_t extraRanks = (int64_t)idxStack.size() - (rank - 2);
    assert(extraRanks >= 0);

    llvm::SmallVector<uint64_t, 6> idx(
        idxStack.begin() + extraRanks, idxStack.end());
    idx.push_back(a);
    idx.push_back(b);

    for (size_t i = 0; i < idx.size() - 2; ++i) {
      uint64_t maxIdx = normShape[i + extraRanks];
      assert(maxIdx > 0);
      --maxIdx;
      idx[i] = std::min(idx[i], maxIdx);
    }
    // Trust that move will be ellided. Don't temper with this statement.
    // Even surrounding `idx` with parenthesis will break it.
    // Go figure. https://godbolt.org/z/dxbeM9jdK
    return idx;
  }

  static std::vector<APInt> createInitialResultStorage(ShapedType resultType) {
    IntegerType resultElementType =
        resultType.getElementType().cast<IntegerType>();
    const APInt emptyInt(
        resultElementType.getWidth(), 0, !resultElementType.isUnsigned());
    std::vector<APInt> resultAttr(resultType.getNumElements(), emptyInt);
    return std::move(resultAttr);
  }

  static std::vector<size_t> calculateDimOffsets(
      ArrayRef<int64_t> resultShapeRef) {
    std::vector<int64_t> resultShape(resultShapeRef);
    std::reverse(resultShape.begin(), resultShape.end());
    std::vector<size_t> offsets;

    size_t accumulator = 1;
    for (int64_t dimSize : resultShape) {
      accumulator *= dimSize;
      offsets.push_back(accumulator);
    }
    std::reverse(offsets.begin(), offsets.end());
    return std::move(offsets);
  }

  const Value _resultValue;
  const ShapedType _resultType;
  const ArrayRef<int64_t> _resultShape;
  const DenseElementsAttr _lhs;
  const DenseElementsAttr _rhs;
  const std::vector<int64_t> _lhsShape;
  const std::vector<int64_t> _rhsShape;
  const std::vector<size_t> _dimOffsets;
  const int64_t _M, _K, _N;
  std::vector<APInt> _resultStorage;
  std::vector<int64_t> _idxStack;
};
} // namespace

DenseElementsAttr CreateMatMulIntegerOfConsts(
    Builder &builder, Value resultValue, Attribute _lhs, Attribute _rhs) {
  // Get the shape vectors of both operands and reverse them
  DenseElementsAttr lhs = _lhs.cast<DenseElementsAttr>();
  DenseElementsAttr rhs = _rhs.cast<DenseElementsAttr>();
  std::vector<int64_t> lhsShape(lhs.getType().getShape());
  std::reverse(lhsShape.begin(), lhsShape.end());
  std::vector<int64_t> rhsShape(rhs.getType().getShape());
  std::reverse(rhsShape.begin(), rhsShape.end());
  const size_t resultNumRanks = std::max(lhsShape.size(), rhsShape.size());
  // Extend the shorter shape with 1 to make them the same length, and reverse
  // back
  lhsShape.resize(resultNumRanks, 1);
  std::reverse(lhsShape.begin(), lhsShape.end());
  rhsShape.resize(resultNumRanks, 1);
  std::reverse(rhsShape.begin(), rhsShape.end());

  ShapedType resultType = resultValue.getType().cast<ShapedType>();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  assert(resultShape.size() == resultNumRanks);

  for (size_t i = 0; i < resultNumRanks - 2; ++i) {
    size_t lhsDim = lhsShape.at(i);
    size_t rhsDim = rhsShape.at(i);
    size_t expectedDim = std::max(lhsDim, rhsDim);
    assert(lhsDim == rhsDim || lhsDim == 1 || rhsDim == 1);
    assert(resultShape[i] == expectedDim && "Unexpected broadcast shape");
  }

  int64_t m = lhsShape.at(lhsShape.size() - 2);
  int64_t k = lhsShape.back();
  int64_t n = rhsShape.back();

  BroadcastMatMulCalculator calculator(resultValue, resultType, resultShape,
      lhs, rhs, std::move(lhsShape), std::move(rhsShape), m, k, n);
  std::vector<APInt> result = calculator.calc();
  return DenseElementsAttr::get(
      resultValue.getType().cast<ShapedType>(), result);
}
