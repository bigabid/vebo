// Mock API functions for AWS Athena Explorer
// These would be replaced with actual API calls to your backend

export interface Table {
  name: string;
}

export interface PartitionData {
  partitionKeys: string[];
  partitions: Record<string, string>[];
}

export interface InsightsJob {
  jobId: string;
  status: 'running' | 'complete' | 'error';
  progress?: number;
  message?: string;
  insights?: InsightsData;
}

export interface InsightsData {
  table: string;
  appliedFilters: Record<string, string[]>;
  rowCount: number;
  partitionSummary: {
    partitionKeys: string[];
    selectedCount: number;
    totalDistinct: number;
  };
  columns: ColumnInsight[];
}

export interface ColumnInsight {
  name: string;
  type: string;
  nullRatio: number;
  topValues?: { value: string; count: number }[];
  numeric?: { min: number; max: number; avg: number };
  temporal?: { min: string; max: string };
}

// Mock delay utility
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Mock data
const MOCK_TABLES: Table[] = [
  { name: 'events_prod' },
  { name: 'purchases_2025' },
  { name: 'impressions_daily' },
  { name: 'user_sessions' },
  { name: 'product_analytics' }
];

const MOCK_PARTITIONS: Record<string, PartitionData> = {
  'events_prod': {
    partitionKeys: ['dt', 'country'],
    partitions: [
      { dt: '2025-08-30', country: 'US' },
      { dt: '2025-08-31', country: 'US' },
      { dt: '2025-09-01', country: 'US' },
      { dt: '2025-08-30', country: 'IL' },
      { dt: '2025-08-31', country: 'IL' },
      { dt: '2025-09-01', country: 'IL' },
      { dt: '2025-08-30', country: 'CA' },
      { dt: '2025-08-31', country: 'CA' }
    ]
  },
  'purchases_2025': {
    partitionKeys: ['year', 'month'],
    partitions: [
      { year: '2025', month: '08' },
      { year: '2025', month: '09' },
      { year: '2025', month: '10' }
    ]
  },
  'impressions_daily': {
    partitionKeys: [],
    partitions: []
  }
};

const MOCK_INSIGHTS: InsightsData = {
  table: 'events_prod',
  appliedFilters: { dt: ['2025-08-31', '2025-09-01'], country: ['US', 'IL'] },
  rowCount: 1234567,
  partitionSummary: {
    partitionKeys: ['dt', 'country'],
    selectedCount: 4,
    totalDistinct: 120
  },
  columns: [
    {
      name: 'user_id',
      type: 'string',
      nullRatio: 0.01,
      topValues: [
        { value: 'u_123', count: 4321 },
        { value: 'u_456', count: 3987 },
        { value: 'u_789', count: 3654 },
        { value: 'u_012', count: 3201 },
        { value: 'u_345', count: 2876 }
      ]
    },
    {
      name: 'revenue',
      type: 'double',
      nullRatio: 0.10,
      numeric: { min: 0.0, max: 999.99, avg: 34.52 }
    },
    {
      name: 'event_time',
      type: 'timestamp',
      nullRatio: 0.00,
      temporal: { min: '2025-08-31T00:00:00Z', max: '2025-09-01T23:59:59Z' }
    },
    {
      name: 'category',
      type: 'string',
      nullRatio: 0.05,
      topValues: [
        { value: 'electronics', count: 125000 },
        { value: 'clothing', count: 98000 },
        { value: 'books', count: 87000 },
        { value: 'home', count: 76000 },
        { value: 'sports', count: 65000 }
      ]
    },
    {
      name: 'session_duration',
      type: 'integer',
      nullRatio: 0.02,
      numeric: { min: 1, max: 7200, avg: 245 }
    }
  ]
};

// API Functions
export const fetchTables = async (): Promise<{ tables: Table[] }> => {
  await delay(800); // Simulate network delay
  return { tables: MOCK_TABLES };
};

export const fetchPartitions = async (table: string): Promise<PartitionData> => {
  await delay(600);
  return MOCK_PARTITIONS[table] || { partitionKeys: [], partitions: [] };
};

export const startInsightsJob = async (
  table: string,
  partitions: Record<string, string[]>
): Promise<{ jobId: string; status: string }> => {
  await delay(400);
  const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  return { jobId, status: 'running' };
};

export const getJobStatus = async (jobId: string): Promise<InsightsJob> => {
  await delay(300);
  
  // Simulate job progression
  const progress = Math.min(100, (Date.now() % 10000) / 100);
  
  if (progress < 90) {
    return {
      jobId,
      status: 'running',
      progress: Math.floor(progress)
    };
  }
  
  return {
    jobId,
    status: 'complete',
    insights: MOCK_INSIGHTS
  };
};