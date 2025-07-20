using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using SqrtSpace.SpaceTime.Linq;

namespace SqrtSpace.SpaceTime.Examples
{
    /// <summary>
    /// Examples demonstrating SpaceTime optimizations for C# developers
    /// </summary>
    public class SpaceTimeExamples
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("SpaceTime LINQ Extensions - C# Examples");
            Console.WriteLine("======================================\n");

            // Example 1: Large data sorting
            SortingExample();
            
            // Example 2: Memory-efficient grouping
            GroupingExample();
            
            // Example 3: Checkpointed processing
            CheckpointExample();
            
            // Example 4: Real-world e-commerce scenario
            await ECommerceExample();
            
            // Example 5: Log file analysis
            LogAnalysisExample();

            Console.WriteLine("\nAll examples completed!");
        }

        /// <summary>
        /// Example 1: Sorting large datasets with minimal memory
        /// </summary>
        private static void SortingExample()
        {
            Console.WriteLine("Example 1: Sorting 10 million items");
            Console.WriteLine("-----------------------------------");

            // Generate large dataset
            var random = new Random(42);
            var largeData = Enumerable.Range(0, 10_000_000)
                .Select(i => new Order 
                { 
                    Id = i, 
                    Total = (decimal)(random.NextDouble() * 1000),
                    Date = DateTime.Now.AddDays(-random.Next(365))
                });

            var sw = Stopwatch.StartNew();
            var memoryBefore = GC.GetTotalMemory(true);

            // Standard LINQ (loads all into memory)
            Console.WriteLine("Standard LINQ OrderBy:");
            var standardSorted = largeData.OrderBy(o => o.Total).Take(100).ToList();
            
            var standardTime = sw.Elapsed;
            var standardMemory = GC.GetTotalMemory(false) - memoryBefore;
            Console.WriteLine($"  Time: {standardTime.TotalSeconds:F2}s");
            Console.WriteLine($"  Memory: {standardMemory / 1_048_576:F1} MB");

            // Reset
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            
            sw.Restart();
            memoryBefore = GC.GetTotalMemory(true);

            // SpaceTime LINQ (√n memory)
            Console.WriteLine("\nSpaceTime OrderByExternal:");
            var sqrtSorted = largeData.OrderByExternal(o => o.Total).Take(100).ToList();
            
            var sqrtTime = sw.Elapsed;
            var sqrtMemory = GC.GetTotalMemory(false) - memoryBefore;
            Console.WriteLine($"  Time: {sqrtTime.TotalSeconds:F2}s");
            Console.WriteLine($"  Memory: {sqrtMemory / 1_048_576:F1} MB");
            Console.WriteLine($"  Memory reduction: {(1 - (double)sqrtMemory / standardMemory) * 100:F1}%");
            Console.WriteLine($"  Time overhead: {(sqrtTime.TotalSeconds / standardTime.TotalSeconds - 1) * 100:F1}%\n");
        }

        /// <summary>
        /// Example 2: Grouping with external memory
        /// </summary>
        private static void GroupingExample()
        {
            Console.WriteLine("Example 2: Grouping customers by region");
            Console.WriteLine("--------------------------------------");

            // Simulate customer data
            var customers = GenerateCustomers(1_000_000);

            var sw = Stopwatch.StartNew();
            var memoryBefore = GC.GetTotalMemory(true);

            // SpaceTime grouping with √n memory
            var groupedByRegion = customers
                .GroupByExternal(c => c.Region)
                .Select(g => new 
                { 
                    Region = g.Key, 
                    Count = g.Count(),
                    TotalRevenue = g.Sum(c => c.TotalPurchases)
                })
                .ToList();

            sw.Stop();
            var memory = GC.GetTotalMemory(false) - memoryBefore;

            Console.WriteLine($"Grouped {customers.Count():N0} customers into {groupedByRegion.Count} regions");
            Console.WriteLine($"Time: {sw.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"Memory used: {memory / 1_048_576:F1} MB");
            Console.WriteLine($"Top regions:");
            foreach (var region in groupedByRegion.OrderByDescending(r => r.Count).Take(5))
            {
                Console.WriteLine($"  {region.Region}: {region.Count:N0} customers, ${region.TotalRevenue:N2} revenue");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 3: Fault-tolerant processing with checkpoints
        /// </summary>
        private static void CheckpointExample()
        {
            Console.WriteLine("Example 3: Processing with checkpoints");
            Console.WriteLine("-------------------------------------");

            var data = Enumerable.Range(0, 100_000)
                .Select(i => new ComputeTask { Id = i, Input = i * 2.5 });

            var sw = Stopwatch.StartNew();
            
            // Process with automatic √n checkpointing
            var results = data
                .Select(task => new ComputeResult 
                { 
                    Id = task.Id, 
                    Output = ExpensiveComputation(task.Input) 
                })
                .ToCheckpointedList();

            sw.Stop();
            
            Console.WriteLine($"Processed {results.Count:N0} tasks in {sw.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"Checkpoints were created every {Math.Sqrt(results.Count):F0} items");
            Console.WriteLine("If the process had failed, it would resume from the last checkpoint\n");
        }

        /// <summary>
        /// Example 4: Real-world e-commerce order processing
        /// </summary>
        private static async Task ECommerceExample()
        {
            Console.WriteLine("Example 4: E-commerce order processing pipeline");
            Console.WriteLine("----------------------------------------------");

            // Simulate order stream
            var orderStream = GenerateOrderStreamAsync(50_000);

            var processedCount = 0;
            var totalRevenue = 0m;

            // Process orders in √n batches for optimal memory usage
            await foreach (var batch in orderStream.BufferAsync())
            {
                // Process batch
                var batchResults = batch
                    .Where(o => o.Status == OrderStatus.Pending)
                    .Select(o => ProcessOrder(o))
                    .ToList();

                // Update metrics
                processedCount += batchResults.Count;
                totalRevenue += batchResults.Sum(o => o.Total);

                // Simulate batch completion
                if (processedCount % 10000 == 0)
                {
                    Console.WriteLine($"  Processed {processedCount:N0} orders, Revenue: ${totalRevenue:N2}");
                }
            }

            Console.WriteLine($"Total: {processedCount:N0} orders, ${totalRevenue:N2} revenue\n");
        }

        /// <summary>
        /// Example 5: Log file analysis with external memory
        /// </summary>
        private static void LogAnalysisExample()
        {
            Console.WriteLine("Example 5: Analyzing large log files");
            Console.WriteLine("-----------------------------------");

            // Simulate log entries
            var logEntries = GenerateLogEntries(5_000_000);

            var sw = Stopwatch.StartNew();

            // Find unique IPs using external distinct
            var uniqueIPs = logEntries
                .Select(e => e.IPAddress)
                .DistinctExternal(maxMemoryItems: 10_000)  // Only keep 10K IPs in memory
                .Count();

            // Find top error codes with memory-efficient grouping
            var topErrors = logEntries
                .Where(e => e.Level == "ERROR")
                .GroupByExternal(e => e.ErrorCode)
                .Select(g => new { ErrorCode = g.Key, Count = g.Count() })
                .OrderByExternal(e => e.Count)
                .TakeLast(10)
                .ToList();

            sw.Stop();

            Console.WriteLine($"Analyzed {5_000_000:N0} log entries in {sw.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"Found {uniqueIPs:N0} unique IP addresses");
            Console.WriteLine("Top error codes:");
            foreach (var error in topErrors.OrderByDescending(e => e.Count))
            {
                Console.WriteLine($"  {error.ErrorCode}: {error.Count:N0} occurrences");
            }
            Console.WriteLine();
        }

        // Helper methods and classes

        private static double ExpensiveComputation(double input)
        {
            // Simulate expensive computation
            return Math.Sqrt(Math.Sin(input) * Math.Cos(input) + 1);
        }

        private static Order ProcessOrder(Order order)
        {
            // Simulate order processing
            order.Status = OrderStatus.Processed;
            order.ProcessedAt = DateTime.UtcNow;
            return order;
        }

        private static IEnumerable<Customer> GenerateCustomers(int count)
        {
            var random = new Random(42);
            var regions = new[] { "North", "South", "East", "West", "Central" };

            for (int i = 0; i < count; i++)
            {
                yield return new Customer
                {
                    Id = i,
                    Name = $"Customer_{i}",
                    Region = regions[random.Next(regions.Length)],
                    TotalPurchases = (decimal)(random.NextDouble() * 10000)
                };
            }
        }

        private static async IAsyncEnumerable<Order> GenerateOrderStreamAsync(int count)
        {
            var random = new Random(42);
            
            for (int i = 0; i < count; i++)
            {
                yield return new Order
                {
                    Id = i,
                    Total = (decimal)(random.NextDouble() * 500),
                    Date = DateTime.Now,
                    Status = OrderStatus.Pending
                };

                // Simulate streaming delay
                if (i % 1000 == 0)
                {
                    await Task.Delay(1);
                }
            }
        }

        private static IEnumerable<LogEntry> GenerateLogEntries(int count)
        {
            var random = new Random(42);
            var levels = new[] { "INFO", "WARN", "ERROR", "DEBUG" };
            var errorCodes = new[] { "404", "500", "503", "400", "401", "403" };

            for (int i = 0; i < count; i++)
            {
                var level = levels[random.Next(levels.Length)];
                yield return new LogEntry
                {
                    Timestamp = DateTime.Now.AddSeconds(-i),
                    Level = level,
                    IPAddress = $"192.168.{random.Next(256)}.{random.Next(256)}",
                    ErrorCode = level == "ERROR" ? errorCodes[random.Next(errorCodes.Length)] : null,
                    Message = $"Log entry {i}"
                };
            }
        }

        // Data classes

        private class Order
        {
            public int Id { get; set; }
            public decimal Total { get; set; }
            public DateTime Date { get; set; }
            public OrderStatus Status { get; set; }
            public DateTime? ProcessedAt { get; set; }
        }

        private enum OrderStatus
        {
            Pending,
            Processed,
            Shipped,
            Delivered
        }

        private class Customer
        {
            public int Id { get; set; }
            public string Name { get; set; }
            public string Region { get; set; }
            public decimal TotalPurchases { get; set; }
        }

        private class ComputeTask
        {
            public int Id { get; set; }
            public double Input { get; set; }
        }

        private class ComputeResult
        {
            public int Id { get; set; }
            public double Output { get; set; }
        }

        private class LogEntry
        {
            public DateTime Timestamp { get; set; }
            public string Level { get; set; }
            public string IPAddress { get; set; }
            public string ErrorCode { get; set; }
            public string Message { get; set; }
        }
    }

    /// <summary>
    /// Benchmarks comparing standard LINQ vs SpaceTime LINQ
    /// </summary>
    public class SpaceTimeBenchmarks
    {
        public static void RunBenchmarks()
        {
            Console.WriteLine("SpaceTime LINQ Benchmarks");
            Console.WriteLine("========================\n");

            // Benchmark 1: Sorting
            BenchmarkSorting();

            // Benchmark 2: Grouping
            BenchmarkGrouping();

            // Benchmark 3: Distinct
            BenchmarkDistinct();

            // Benchmark 4: Join
            BenchmarkJoin();
        }

        private static void BenchmarkSorting()
        {
            Console.WriteLine("Benchmark: Sorting Performance");
            Console.WriteLine("-----------------------------");

            var sizes = new[] { 10_000, 100_000, 1_000_000 };

            foreach (var size in sizes)
            {
                var data = Enumerable.Range(0, size)
                    .Select(i => new { Id = i, Value = Random.Shared.NextDouble() })
                    .ToList();

                // Standard LINQ
                GC.Collect();
                var memBefore = GC.GetTotalMemory(true);
                var sw = Stopwatch.StartNew();
                
                var standardResult = data.OrderBy(x => x.Value).ToList();
                
                var standardTime = sw.Elapsed;
                var standardMem = GC.GetTotalMemory(false) - memBefore;

                // SpaceTime LINQ
                GC.Collect();
                memBefore = GC.GetTotalMemory(true);
                sw.Restart();
                
                var sqrtResult = data.OrderByExternal(x => x.Value).ToList();
                
                var sqrtTime = sw.Elapsed;
                var sqrtMem = GC.GetTotalMemory(false) - memBefore;

                Console.WriteLine($"\nSize: {size:N0}");
                Console.WriteLine($"  Standard: {standardTime.TotalMilliseconds:F0}ms, {standardMem / 1_048_576.0:F1}MB");
                Console.WriteLine($"  SpaceTime: {sqrtTime.TotalMilliseconds:F0}ms, {sqrtMem / 1_048_576.0:F1}MB");
                Console.WriteLine($"  Memory saved: {(1 - (double)sqrtMem / standardMem) * 100:F1}%");
                Console.WriteLine($"  Time overhead: {(sqrtTime.TotalMilliseconds / standardTime.TotalMilliseconds - 1) * 100:F1}%");
            }
            Console.WriteLine();
        }

        private static void BenchmarkGrouping()
        {
            Console.WriteLine("Benchmark: Grouping Performance");
            Console.WriteLine("------------------------------");

            var size = 1_000_000;
            var data = Enumerable.Range(0, size)
                .Select(i => new { Id = i, Category = $"Cat_{i % 100}" })
                .ToList();

            // Standard LINQ
            GC.Collect();
            var sw = Stopwatch.StartNew();
            var standardGroups = data.GroupBy(x => x.Category).ToList();
            var standardTime = sw.Elapsed;

            // SpaceTime LINQ
            GC.Collect();
            sw.Restart();
            var sqrtGroups = data.GroupByExternal(x => x.Category).ToList();
            var sqrtTime = sw.Elapsed;

            Console.WriteLine($"Grouped {size:N0} items into {standardGroups.Count} groups");
            Console.WriteLine($"  Standard: {standardTime.TotalMilliseconds:F0}ms");
            Console.WriteLine($"  SpaceTime: {sqrtTime.TotalMilliseconds:F0}ms");
            Console.WriteLine($"  Time ratio: {sqrtTime.TotalMilliseconds / standardTime.TotalMilliseconds:F2}x\n");
        }

        private static void BenchmarkDistinct()
        {
            Console.WriteLine("Benchmark: Distinct Performance");
            Console.WriteLine("------------------------------");

            var size = 5_000_000;
            var uniqueCount = 100_000;
            var data = Enumerable.Range(0, size)
                .Select(i => i % uniqueCount)
                .ToList();

            // Standard LINQ
            GC.Collect();
            var memBefore = GC.GetTotalMemory(true);
            var sw = Stopwatch.StartNew();
            
            var standardDistinct = data.Distinct().Count();
            
            var standardTime = sw.Elapsed;
            var standardMem = GC.GetTotalMemory(false) - memBefore;

            // SpaceTime LINQ
            GC.Collect();
            memBefore = GC.GetTotalMemory(true);
            sw.Restart();
            
            var sqrtDistinct = data.DistinctExternal(maxMemoryItems: 10_000).Count();
            
            var sqrtTime = sw.Elapsed;
            var sqrtMem = GC.GetTotalMemory(false) - memBefore;

            Console.WriteLine($"Found {standardDistinct:N0} unique items in {size:N0} total");
            Console.WriteLine($"  Standard: {standardTime.TotalMilliseconds:F0}ms, {standardMem / 1_048_576.0:F1}MB");
            Console.WriteLine($"  SpaceTime: {sqrtTime.TotalMilliseconds:F0}ms, {sqrtMem / 1_048_576.0:F1}MB");
            Console.WriteLine($"  Memory saved: {(1 - (double)sqrtMem / standardMem) * 100:F1}%\n");
        }

        private static void BenchmarkJoin()
        {
            Console.WriteLine("Benchmark: Join Performance");
            Console.WriteLine("--------------------------");

            var outerSize = 100_000;
            var innerSize = 50_000;
            
            var customers = Enumerable.Range(0, outerSize)
                .Select(i => new { CustomerId = i, Name = $"Customer_{i}" })
                .ToList();
                
            var orders = Enumerable.Range(0, innerSize)
                .Select(i => new { OrderId = i, CustomerId = i % outerSize, Total = i * 10.0 })
                .ToList();

            // Standard LINQ
            GC.Collect();
            var sw = Stopwatch.StartNew();
            
            var standardJoin = customers.Join(orders,
                c => c.CustomerId,
                o => o.CustomerId,
                (c, o) => new { c.Name, o.Total })
                .Count();
                
            var standardTime = sw.Elapsed;

            // SpaceTime LINQ
            GC.Collect();
            sw.Restart();
            
            var sqrtJoin = customers.JoinExternal(orders,
                c => c.CustomerId,
                o => o.CustomerId,
                (c, o) => new { c.Name, o.Total })
                .Count();
                
            var sqrtTime = sw.Elapsed;

            Console.WriteLine($"Joined {outerSize:N0} customers with {innerSize:N0} orders");
            Console.WriteLine($"  Standard: {standardTime.TotalMilliseconds:F0}ms");
            Console.WriteLine($"  SpaceTime: {sqrtTime.TotalMilliseconds:F0}ms");
            Console.WriteLine($"  Time ratio: {sqrtTime.TotalMilliseconds / standardTime.TotalMilliseconds:F2}x\n");
        }
    }
}