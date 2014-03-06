/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linalg

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import org.jblas.{DoubleMatrix, Singular, MatrixFunctions}


/**
 * Class used to obtain singular value decompositions
 */
class PCA {
  private var k: Int = 1

  /**
   * Set the number of top-k singular vectors to return
   */
  def setK(k: Int): PCA = {
    this.k = k
    this
  }

   /**
   * Compute PCA using the current set parameters
   */
  def compute(matrix: SparseMatrix) : SparseMatrix = {
    PCA.computePCA(matrix, k)
  }
}


/**
 * Top-level methods for calling Singular Value Decomposition
 * NOTE: All matrices are in 0-indexed sparse format RDD[((int, int), value)]
 */
object PCA {
/**
 * Singular Value Decomposition for Tall and Skinny matrices.
 * Given an m x n matrix A, this will compute matrices U, S, V such that
 * A = U * S * V'
 * 
 * There is no restriction on m, but we require n^2 doubles to fit in memory.
 * Further, n should be less than m.
 * 
 * The decomposition is computed by first computing A'A = V S^2 V',
 * computing svd locally on that (since n x n is small),
 * from which we recover S and V. 
 * Then we compute U via easy matrix multiplication
 * as U =  A * V * S^-1
 * 
 * Only the k largest singular values and associated vectors are found.
 * If there are k such values, then the dimensions of the return will be:
 *
 * S is k x k and diagonal, holding the singular values on diagonal
 * U is m x k and satisfies U'U = eye(k)
 * V is n x k and satisfies V'V = eye(k)
 *
 * All input and output is expected in sparse matrix format, 0-indexed
 * as tuples of the form ((i,j),value) all in RDDs using the
 * SparseMatrix class
 *
 * @param matrix sparse matrix to factorize
 * @param k Recover k singular values and vectors
 * @return Three sparse matrices: U, S, V such that A = USV^T
 */
  def computePCA(
      matrix: SparseMatrix,
      k: Int)
    : SparseMatrix =
  {
    val rawdata = matrix.data
    val m = matrix.m
    val n = matrix.n

    if (m <= 0 || n <= 0) {
      throw new IllegalArgumentException("Expecting a well-formed matrix")
    }

    // compute column sums and normalize matrix
    val colsums = rawdata.map(entry => (entry.j, entry.mval)).reduceByKey(_+_)
    val data = rawdata.map(entry => (entry.j, (entry.i, entry.mval))).join(colsums).map{
      case (col, ((row, mval), colsum)) =>
        MatrixEntry(row, col, (mval - colsum / m.toDouble) / Math.sqrt(n-1)) }


    val mysvd = new SVD
    val retV = mysvd.setK(k).computeU(false).compute(SparseMatrix(data, m, n)).V
    retV
  }


  def main(args: Array[String]) {
    if (args.length < 6) {
      println("Usage: PCA <master> <matrix_file> <m> <n> " +
              "<k> <output_coefficient_file>")
      System.exit(1)
    }

    val (master, inputFile, m, n, k, output_u) = 
      (args(0), args(1), args(2).toInt, args(3).toInt,
      args(4).toInt, args(5))
    
    val sc = new SparkContext(master, "PCA")
    
    val rawdata = sc.textFile(inputFile)
    val data = rawdata.map { line =>
      val parts = line.split(',')
      MatrixEntry(parts(0).toInt, parts(1).toInt, parts(2).toDouble)
    }

    val u = PCA.computePCA(SparseMatrix(data, m, n), k)
    
    println("Computed " + k + " principal vectors")
    u.data.saveAsTextFile(output_u)
    System.exit(0)
  }
}


