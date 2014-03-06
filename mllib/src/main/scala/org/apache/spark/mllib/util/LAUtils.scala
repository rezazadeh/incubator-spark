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

package org.apache.spark.mllib.util

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.linalg._

/**
 * Helper methods for linear algebra
 */
object LAUtils {
  /**
   * Convert a SparseMatrix into a DenseMatrix
   *
   * @param sp Sparse matrix to be converted
   * @return dense version of the input
   */
  def spToDense(sp: SparseMatrix): DenseMatrix = {
    val m = sp.m
    val n = sp.n
    val rows = sp.data.map(x => (x.i, (x.j, x.mval))).groupByKey.map{
      case (i, cols) =>
        val rowarray = Array.ofDim[Double](n)
        for (i <- 0 to cols.size - 1) {
          rowarray(cols(i)._1) = cols(i)._2 
        }
        MatrixRow(i, rowarray)
    }
    DenseMatrix(rows, m, n)
  }

  /**
   * Convert a DenseMatrix to a SparseMatrix
   *
   * @param a matrix to be converted
   * @return sparse version of the input
   */
  def denseToSp(a: DenseMatrix): SparseMatrix = {
    val m = a.m
    val n = a.n
    val data = a.rows.flatMap{
      mrow => Array.tabulate(n)(j => MatrixEntry(mrow.i, j, mrow.data(j)))
    }
    SparseMatrix(data, m, n)
  }
}
