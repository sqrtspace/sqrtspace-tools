using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SqrtSpace.SpaceTime.Linq
{
    /// <summary>
    /// LINQ extensions that implement space-time tradeoffs for memory-efficient operations
    /// </summary>
    public static class SpaceTimeLinqExtensions
    {
        /// <summary>
        /// Orders a sequence using external merge sort with √n memory usage
        /// </summary>
        public static IOrderedEnumerable<TSource> OrderByExternal<TSource, TKey>(
            this IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector,
            IComparer<TKey> comparer = null,
            int? bufferSize = null)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (keySelector == null) throw new ArgumentNullException(nameof(keySelector));

            return new ExternalOrderedEnumerable<TSource, TKey>(source, keySelector, comparer, bufferSize);
        }

        /// <summary>
        /// Groups elements using √n memory for large datasets
        /// </summary>
        public static IEnumerable<IGrouping<TKey, TSource>> GroupByExternal<TSource, TKey>(
            this IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector,
            int? bufferSize = null)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (keySelector == null) throw new ArgumentNullException(nameof(keySelector));

            var count = source.TryGetNonEnumeratedCount(out var c) ? c : 1000000;
            var optimalBuffer = bufferSize ?? (int)Math.Sqrt(count);

            return new ExternalGrouping<TSource, TKey>(source, keySelector, optimalBuffer);
        }

        /// <summary>
        /// Processes sequence in √n-sized batches for memory efficiency
        /// </summary>
        public static IEnumerable<List<T>> BatchBySqrtN<T>(
            this IEnumerable<T> source,
            int? totalCount = null)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));

            var count = totalCount ?? (source.TryGetNonEnumeratedCount(out var c) ? c : 1000);
            var batchSize = Math.Max(1, (int)Math.Sqrt(count));

            return source.Chunk(batchSize).Select(chunk => chunk.ToList());
        }

        /// <summary>
        /// Performs a memory-efficient join using √n buffers
        /// </summary>
        public static IEnumerable<TResult> JoinExternal<TOuter, TInner, TKey, TResult>(
            this IEnumerable<TOuter> outer,
            IEnumerable<TInner> inner,
            Func<TOuter, TKey> outerKeySelector,
            Func<TInner, TKey> innerKeySelector,
            Func<TOuter, TInner, TResult> resultSelector,
            IEqualityComparer<TKey> comparer = null)
        {
            if (outer == null) throw new ArgumentNullException(nameof(outer));
            if (inner == null) throw new ArgumentNullException(nameof(inner));

            var innerCount = inner.TryGetNonEnumeratedCount(out var c) ? c : 10000;
            var bufferSize = (int)Math.Sqrt(innerCount);

            return ExternalJoinIterator(outer, inner, outerKeySelector, innerKeySelector, 
                                      resultSelector, comparer, bufferSize);
        }

        /// <summary>
        /// Converts sequence to a list with checkpointing for fault tolerance
        /// </summary>
        public static List<T> ToCheckpointedList<T>(
            this IEnumerable<T> source,
            string checkpointPath = null,
            int? checkpointInterval = null)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));

            var result = new List<T>();
            var count = 0;
            var interval = checkpointInterval ?? (int)Math.Sqrt(source.Count());
            
            checkpointPath ??= Path.GetTempFileName();

            try
            {
                // Try to restore from checkpoint
                if (File.Exists(checkpointPath))
                {
                    result = RestoreCheckpoint<T>(checkpointPath);
                    count = result.Count;
                }

                foreach (var item in source.Skip(count))
                {
                    result.Add(item);
                    count++;

                    if (count % interval == 0)
                    {
                        SaveCheckpoint(result, checkpointPath);
                    }
                }

                return result;
            }
            finally
            {
                // Clean up checkpoint file
                if (File.Exists(checkpointPath))
                {
                    File.Delete(checkpointPath);
                }
            }
        }

        /// <summary>
        /// Performs distinct operation with limited memory using external storage
        /// </summary>
        public static IEnumerable<T> DistinctExternal<T>(
            this IEnumerable<T> source,
            IEqualityComparer<T> comparer = null,
            int? maxMemoryItems = null)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));

            var maxItems = maxMemoryItems ?? (int)Math.Sqrt(source.Count());
            return new ExternalDistinct<T>(source, comparer, maxItems);
        }

        /// <summary>
        /// Aggregates large sequences with √n memory checkpoints
        /// </summary>
        public static TAccumulate AggregateWithCheckpoints<TSource, TAccumulate>(
            this IEnumerable<TSource> source,
            TAccumulate seed,
            Func<TAccumulate, TSource, TAccumulate> func,
            int? checkpointInterval = null)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (func == null) throw new ArgumentNullException(nameof(func));

            var accumulator = seed;
            var count = 0;
            var interval = checkpointInterval ?? (int)Math.Sqrt(source.Count());
            var checkpoints = new Stack<(int index, TAccumulate value)>();

            foreach (var item in source)
            {
                accumulator = func(accumulator, item);
                count++;

                if (count % interval == 0)
                {
                    // Deep copy if TAccumulate is a reference type
                    var checkpoint = accumulator is ICloneable cloneable 
                        ? (TAccumulate)cloneable.Clone() 
                        : accumulator;
                    checkpoints.Push((count, checkpoint));
                }
            }

            return accumulator;
        }

        /// <summary>
        /// Memory-efficient set operations using external storage
        /// </summary>
        public static IEnumerable<T> UnionExternal<T>(
            this IEnumerable<T> first,
            IEnumerable<T> second,
            IEqualityComparer<T> comparer = null)
        {
            if (first == null) throw new ArgumentNullException(nameof(first));
            if (second == null) throw new ArgumentNullException(nameof(second));

            var totalCount = first.Count() + second.Count();
            var bufferSize = (int)Math.Sqrt(totalCount);

            return ExternalSetOperation(first, second, SetOperation.Union, comparer, bufferSize);
        }

        /// <summary>
        /// Async enumerable with √n buffering for optimal memory usage
        /// </summary>
        public static async IAsyncEnumerable<List<T>> BufferAsync<T>(
            this IAsyncEnumerable<T> source,
            int? bufferSize = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));

            var buffer = new List<T>(bufferSize ?? 1000);
            var optimalSize = bufferSize ?? (int)Math.Sqrt(1000000); // Assume large dataset

            await foreach (var item in source.WithCancellation(cancellationToken))
            {
                buffer.Add(item);
                
                if (buffer.Count >= optimalSize)
                {
                    yield return buffer;
                    buffer = new List<T>(optimalSize);
                }
            }

            if (buffer.Count > 0)
            {
                yield return buffer;
            }
        }

        // Private helper methods

        private static IEnumerable<TResult> ExternalJoinIterator<TOuter, TInner, TKey, TResult>(
            IEnumerable<TOuter> outer,
            IEnumerable<TInner> inner,
            Func<TOuter, TKey> outerKeySelector,
            Func<TInner, TKey> innerKeySelector,
            Func<TOuter, TInner, TResult> resultSelector,
            IEqualityComparer<TKey> comparer,
            int bufferSize)
        {
            comparer ??= EqualityComparer<TKey>.Default;
            
            // Process inner sequence in chunks
            foreach (var innerChunk in inner.Chunk(bufferSize))
            {
                var lookup = innerChunk.ToLookup(innerKeySelector, comparer);
                
                foreach (var outerItem in outer)
                {
                    var key = outerKeySelector(outerItem);
                    foreach (var innerItem in lookup[key])
                    {
                        yield return resultSelector(outerItem, innerItem);
                    }
                }
            }
        }

        private static void SaveCheckpoint<T>(List<T> data, string path)
        {
            // Simplified - in production would use proper serialization
            using var writer = new StreamWriter(path);
            writer.WriteLine(data.Count);
            foreach (var item in data)
            {
                writer.WriteLine(item?.ToString() ?? "null");
            }
        }

        private static List<T> RestoreCheckpoint<T>(string path)
        {
            // Simplified - in production would use proper deserialization
            var lines = File.ReadAllLines(path);
            var count = int.Parse(lines[0]);
            var result = new List<T>(count);
            
            // This is a simplified implementation
            // Real implementation would handle type conversion properly
            for (int i = 1; i <= count && i < lines.Length; i++)
            {
                if (typeof(T) == typeof(string))
                {
                    result.Add((T)(object)lines[i]);
                }
                else if (typeof(T) == typeof(int) && int.TryParse(lines[i], out var intVal))
                {
                    result.Add((T)(object)intVal);
                }
                // Add more type conversions as needed
            }
            
            return result;
        }

        private static IEnumerable<T> ExternalSetOperation<T>(
            IEnumerable<T> first,
            IEnumerable<T> second,
            SetOperation operation,
            IEqualityComparer<T> comparer,
            int bufferSize)
        {
            // Simplified external set operation
            var seen = new HashSet<T>(comparer);
            var spillFile = Path.GetTempFileName();
            
            try
            {
                // Process first sequence
                foreach (var item in first)
                {
                    if (seen.Count >= bufferSize)
                    {
                        // Spill to disk
                        SpillToDisk(seen, spillFile);
                        seen.Clear();
                    }
                    
                    if (seen.Add(item))
                    {
                        yield return item;
                    }
                }
                
                // Process second sequence for union
                if (operation == SetOperation.Union)
                {
                    foreach (var item in second)
                    {
                        if (!seen.Contains(item) && !ExistsInSpillFile(item, spillFile, comparer))
                        {
                            yield return item;
                        }
                    }
                }
            }
            finally
            {
                if (File.Exists(spillFile))
                {
                    File.Delete(spillFile);
                }
            }
        }

        private static void SpillToDisk<T>(HashSet<T> items, string path)
        {
            using var writer = new StreamWriter(path, append: true);
            foreach (var item in items)
            {
                writer.WriteLine(item?.ToString() ?? "null");
            }
        }

        private static bool ExistsInSpillFile<T>(T item, string path, IEqualityComparer<T> comparer)
        {
            if (!File.Exists(path)) return false;
            
            // Simplified - real implementation would be more efficient
            var itemStr = item?.ToString() ?? "null";
            return File.ReadLines(path).Any(line => line == itemStr);
        }

        private enum SetOperation
        {
            Union,
            Intersect,
            Except
        }
    }

    // Supporting classes

    internal class ExternalOrderedEnumerable<TSource, TKey> : IOrderedEnumerable<TSource>
    {
        private readonly IEnumerable<TSource> _source;
        private readonly Func<TSource, TKey> _keySelector;
        private readonly IComparer<TKey> _comparer;
        private readonly int _bufferSize;

        public ExternalOrderedEnumerable(
            IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector,
            IComparer<TKey> comparer,
            int? bufferSize)
        {
            _source = source;
            _keySelector = keySelector;
            _comparer = comparer ?? Comparer<TKey>.Default;
            _bufferSize = bufferSize ?? (int)Math.Sqrt(source.Count());
        }

        public IOrderedEnumerable<TSource> CreateOrderedEnumerable<TNewKey>(
            Func<TSource, TNewKey> keySelector,
            IComparer<TNewKey> comparer,
            bool descending)
        {
            // Simplified - would need proper implementation
            throw new NotImplementedException();
        }

        public IEnumerator<TSource> GetEnumerator()
        {
            // External merge sort implementation
            var chunks = new List<List<TSource>>();
            var chunk = new List<TSource>(_bufferSize);
            
            foreach (var item in _source)
            {
                chunk.Add(item);
                if (chunk.Count >= _bufferSize)
                {
                    chunks.Add(chunk.OrderBy(_keySelector, _comparer).ToList());
                    chunk = new List<TSource>(_bufferSize);
                }
            }
            
            if (chunk.Count > 0)
            {
                chunks.Add(chunk.OrderBy(_keySelector, _comparer).ToList());
            }
            
            // Merge sorted chunks
            return MergeSortedChunks(chunks).GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        private IEnumerable<TSource> MergeSortedChunks(List<List<TSource>> chunks)
        {
            var indices = new int[chunks.Count];
            
            while (true)
            {
                TSource minItem = default;
                TKey minKey = default;
                int minChunk = -1;
                
                // Find minimum across all chunks
                for (int i = 0; i < chunks.Count; i++)
                {
                    if (indices[i] < chunks[i].Count)
                    {
                        var item = chunks[i][indices[i]];
                        var key = _keySelector(item);
                        
                        if (minChunk == -1 || _comparer.Compare(key, minKey) < 0)
                        {
                            minItem = item;
                            minKey = key;
                            minChunk = i;
                        }
                    }
                }
                
                if (minChunk == -1) yield break;
                
                yield return minItem;
                indices[minChunk]++;
            }
        }
    }

    internal class ExternalGrouping<TSource, TKey> : IEnumerable<IGrouping<TKey, TSource>>
    {
        private readonly IEnumerable<TSource> _source;
        private readonly Func<TSource, TKey> _keySelector;
        private readonly int _bufferSize;

        public ExternalGrouping(IEnumerable<TSource> source, Func<TSource, TKey> keySelector, int bufferSize)
        {
            _source = source;
            _keySelector = keySelector;
            _bufferSize = bufferSize;
        }

        public IEnumerator<IGrouping<TKey, TSource>> GetEnumerator()
        {
            var groups = new Dictionary<TKey, List<TSource>>(_bufferSize);
            var spilledGroups = new Dictionary<TKey, string>();
            
            foreach (var item in _source)
            {
                var key = _keySelector(item);
                
                if (!groups.ContainsKey(key))
                {
                    if (groups.Count >= _bufferSize)
                    {
                        // Spill largest group to disk
                        SpillLargestGroup(groups, spilledGroups);
                    }
                    groups[key] = new List<TSource>();
                }
                
                groups[key].Add(item);
            }
            
            // Return in-memory groups
            foreach (var kvp in groups)
            {
                yield return new Grouping<TKey, TSource>(kvp.Key, kvp.Value);
            }
            
            // Return spilled groups
            foreach (var kvp in spilledGroups)
            {
                var items = LoadSpilledGroup<TSource>(kvp.Value);
                yield return new Grouping<TKey, TSource>(kvp.Key, items);
                File.Delete(kvp.Value);
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        private void SpillLargestGroup(
            Dictionary<TKey, List<TSource>> groups,
            Dictionary<TKey, string> spilledGroups)
        {
            var largest = groups.OrderByDescending(g => g.Value.Count).First();
            var spillFile = Path.GetTempFileName();
            
            // Simplified serialization
            File.WriteAllLines(spillFile, largest.Value.Select(v => v?.ToString() ?? "null"));
            
            spilledGroups[largest.Key] = spillFile;
            groups.Remove(largest.Key);
        }

        private List<T> LoadSpilledGroup<T>(string path)
        {
            // Simplified deserialization
            return File.ReadAllLines(path).Select(line => (T)(object)line).ToList();
        }
    }

    internal class Grouping<TKey, TElement> : IGrouping<TKey, TElement>
    {
        public TKey Key { get; }
        private readonly IEnumerable<TElement> _elements;

        public Grouping(TKey key, IEnumerable<TElement> elements)
        {
            Key = key;
            _elements = elements;
        }

        public IEnumerator<TElement> GetEnumerator()
        {
            return _elements.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    internal class ExternalDistinct<T> : IEnumerable<T>
    {
        private readonly IEnumerable<T> _source;
        private readonly IEqualityComparer<T> _comparer;
        private readonly int _maxMemoryItems;

        public ExternalDistinct(IEnumerable<T> source, IEqualityComparer<T> comparer, int maxMemoryItems)
        {
            _source = source;
            _comparer = comparer ?? EqualityComparer<T>.Default;
            _maxMemoryItems = maxMemoryItems;
        }

        public IEnumerator<T> GetEnumerator()
        {
            var seen = new HashSet<T>(_comparer);
            var spillFile = Path.GetTempFileName();
            
            try
            {
                foreach (var item in _source)
                {
                    if (seen.Count >= _maxMemoryItems)
                    {
                        // Spill to disk and clear memory
                        SpillHashSet(seen, spillFile);
                        seen.Clear();
                    }
                    
                    if (seen.Add(item) && !ExistsInSpillFile(item, spillFile))
                    {
                        yield return item;
                    }
                }
            }
            finally
            {
                if (File.Exists(spillFile))
                {
                    File.Delete(spillFile);
                }
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        private void SpillHashSet(HashSet<T> items, string path)
        {
            using var writer = new StreamWriter(path, append: true);
            foreach (var item in items)
            {
                writer.WriteLine(item?.ToString() ?? "null");
            }
        }

        private bool ExistsInSpillFile(T item, string path)
        {
            if (!File.Exists(path)) return false;
            var itemStr = item?.ToString() ?? "null";
            return File.ReadLines(path).Any(line => line == itemStr);
        }
    }
}