import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchDataSources, fetchCatalogs, fetchDatabases, fetchTables, fetchPartitions, startInsightsJob, getJobStatus } from '@/lib/api';
import { useEffect } from 'react';

export const useDataSources = () => {
  return useQuery({
    queryKey: ['athena', 'data-sources'],
    queryFn: fetchDataSources,
    staleTime: 5 * 60 * 1000,
  });
};

export const useCatalogs = (dataSource: string | null) => {
  return useQuery({
    queryKey: ['athena', 'catalogs', dataSource],
    queryFn: () => fetchCatalogs(dataSource!),
    enabled: !!dataSource,
    staleTime: 5 * 60 * 1000,
  });
};

export const useDatabases = (dataSource: string | null, catalog: string | null) => {
  return useQuery({
    queryKey: ['athena', 'databases', dataSource, catalog],
    queryFn: () => fetchDatabases(dataSource!, catalog!),
    enabled: !!dataSource && !!catalog,
    staleTime: 5 * 60 * 1000,
  });
};

export const useTables = (dataSource: string | null, catalog: string | null, database: string | null) => {
  return useQuery({
    queryKey: ['athena', 'tables', dataSource, catalog, database],
    queryFn: () => fetchTables(dataSource!, catalog!, database!),
    enabled: !!dataSource && !!catalog && !!database,
    staleTime: 5 * 60 * 1000,
  });
};

export const usePartitions = (table: string | null, params?: { catalog?: string; database?: string }) => {
  return useQuery({
    queryKey: ['partitions', params?.catalog, params?.database, table],
    queryFn: () => fetchPartitions(table!, { catalog: params?.catalog, database: params?.database }),
    enabled: !!table && !!params?.catalog && !!params?.database,
    staleTime: 5 * 60 * 1000,
  });
};

export const useStartInsights = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (params: { dataSource: string; catalog: string; database: string; table: string; partitions: Record<string, string[]> }) =>
      startInsightsJob(params),
    onSuccess: () => {
      // Invalidate job status queries to trigger polling
      queryClient.invalidateQueries({ queryKey: ['job-status'] });
    },
  });
};

export const useJobStatus = (jobId: string | null, enabled: boolean = true) => {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: ['job-status', jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: !!jobId && enabled,
    refetchInterval: (query) => {
      // Stop polling if job is complete or errored
      if (query.state.data?.status === 'complete' || query.state.data?.status === 'error') {
        return false;
      }
      return 2000; // Poll every 2 seconds
    },
    refetchIntervalInBackground: true,
  });

  // Stop polling when job completes
  useEffect(() => {
    if (query.data?.status === 'complete' || query.data?.status === 'error') {
      queryClient.setQueryData(['job-status', jobId], query.data);
    }
  }, [query.data?.status, queryClient, jobId]);

  return query;
};