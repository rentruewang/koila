# Schemas

The package, schemas, is a collection of metadata describing the 'type' of data.

There are multiple supported types of metadata:

1. Shape: The shape of the tensor in a column.
2. DType: The data type of each element in the column.
3. Device: The device that the tensor lives on.

There are 2 types of schemas: `ColumnSchema` and `TableSchema`, where the latter is a collection of former.
